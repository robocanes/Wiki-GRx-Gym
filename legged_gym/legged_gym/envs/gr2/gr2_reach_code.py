import numpy as np

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym.envs.gr2.gr2_code import GR2


def _quat_from_rpy(rpy):
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]

    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)

    quat = torch.zeros(rpy.shape[0], 4, dtype=rpy.dtype, device=rpy.device)
    quat[:, 0] = sr * cp * cy - cr * sp * sy
    quat[:, 1] = cr * sp * cy + sr * cp * sy
    quat[:, 2] = cr * cp * sy - sr * sp * cy
    quat[:, 3] = cr * cp * cy + sr * sp * sy
    return quat


def _quat_mul(q, r):
    quat = torch.zeros_like(q)
    quat[:, 0] = q[:, 3] * r[:, 0] + q[:, 0] * r[:, 3] + q[:, 1] * r[:, 2] - q[:, 2] * r[:, 1]
    quat[:, 1] = q[:, 3] * r[:, 1] - q[:, 0] * r[:, 2] + q[:, 1] * r[:, 3] + q[:, 2] * r[:, 0]
    quat[:, 2] = q[:, 3] * r[:, 2] + q[:, 0] * r[:, 1] - q[:, 1] * r[:, 0] + q[:, 2] * r[:, 3]
    quat[:, 3] = q[:, 3] * r[:, 3] - q[:, 0] * r[:, 0] - q[:, 1] * r[:, 1] - q[:, 2] * r[:, 2]
    return quat


class GR2Reach(GR2):
    def _init_buffers_others(self):
        super()._init_buffers_others()

        self.reach_target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.reach_target_pos_base = torch.zeros_like(self.reach_target_pos)
        self.reach_target_rpy_base = torch.zeros_like(self.reach_target_pos)
        self.reach_target_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.reach_use_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reach_arm_selector = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)

        action_dof_names = [self.dof_names[i] for i in self.action_indices.tolist()]
        arm_action_names = ("shoulder", "elbow", "wrist")
        self.left_arm_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if name.startswith("left_") and any(arm_name in name for arm_name in arm_action_names)
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.right_arm_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if name.startswith("right_") and any(arm_name in name for arm_name in arm_action_names)
            ],
            dtype=torch.long,
            device=self.device,
        )

        self._sample_reach_targets(torch.arange(self.num_envs, device=self.device))

    def compute_observation_variables(self):
        super().compute_observation_variables()
        self._update_reach_target_base_frame()

    def compute_observation_profile(self):
        obs_buf = torch.cat(
            (
                self.reach_target_pos_base,
                self.reach_target_rpy_base,
                self.reach_arm_selector,

                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity * self.obs_scales.gravity,

                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,

                self.actions * self.obs_scales.action,
            ), dim=-1)

        pri_obs_buf = torch.cat(
            (
                obs_buf,

                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset * self.obs_scales.height_measurements,

                self.feet_contact,
                self.feet_height * self.obs_scales.height_measurements,
                self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,

                self.surround_heights_offset * self.obs_scales.height_measurements,
            ), dim=-1)

        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def compute_obs_noise_scale_vec_profile(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])

        target_pos_slice = slice(0, 3)
        target_rpy_slice = slice(3, 6)
        noise_vec[target_pos_slice] = self.cfg.noise.noise_scales.target_pos * self.noise_level
        noise_vec[target_rpy_slice] = self.cfg.noise.noise_scales.target_rpy * self.noise_level

        start_index_of_base_related = 8
        noise_vec[start_index_of_base_related + 0:
                  start_index_of_base_related + 3] = \
            self.noise_scales.ang_vel \
            * self.noise_level \
            * self.obs_scales.ang_vel
        noise_vec[start_index_of_base_related + 3:
                  start_index_of_base_related + 6] = \
            self.noise_scales.gravity \
            * self.noise_level \
            * self.obs_scales.gravity

        start_index_of_dof_related = start_index_of_base_related + 6
        noise_vec[start_index_of_dof_related + 0 * self.num_dofs:
                  start_index_of_dof_related + 1 * self.num_dofs] = \
            self.noise_scales.dof_pos \
            * self.noise_level \
            * self.obs_scales.dof_pos
        noise_vec[start_index_of_dof_related + 1 * self.num_dofs:
                  start_index_of_dof_related + 2 * self.num_dofs] = \
            self.noise_scales.dof_vel \
            * self.noise_level \
            * self.obs_scales.dof_vel

        start_index_of_action_related = start_index_of_dof_related + 2 * self.num_dofs
        noise_vec[start_index_of_action_related:
                  start_index_of_action_related + self.num_actions] = \
            self.noise_scales.action \
            * self.noise_level \
            * self.obs_scales.action

        return noise_vec

    def _resample_commands(self, env_ids=None, command_profile=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        self.commands[env_ids] = 0.0
        self._sample_reach_targets(env_ids)

    def _sample_reach_targets(self, env_ids):
        x_range = self.cfg.reach.target_x_range
        y_abs_range = self.cfg.reach.target_y_abs_range
        z_range = self.cfg.reach.target_z_range
        roll_range = self.cfg.reach.target_roll_range
        pitch_range = self.cfg.reach.target_pitch_range
        yaw_range = self.cfg.reach.target_yaw_range

        target_pos_base = torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device)
        target_pos_base[:, 0] = torch_rand_float(x_range[0], x_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        target_y_abs = torch_rand_float(y_abs_range[0], y_abs_range[1], (len(env_ids), 1), device=self.device).squeeze(1)
        target_side = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            torch.ones(len(env_ids), device=self.device),
            -torch.ones(len(env_ids), device=self.device),
        )
        target_pos_base[:, 1] = target_y_abs * target_side
        target_pos_base[:, 2] = torch_rand_float(z_range[0], z_range[1], (len(env_ids), 1), device=self.device).squeeze(1)

        target_rpy_base = torch.zeros_like(target_pos_base)
        target_rpy_base[:, 0] = torch_rand_float(
            roll_range[0], roll_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        target_rpy_base[:, 1] = torch_rand_float(
            pitch_range[0], pitch_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        target_rpy_base[:, 2] = torch_rand_float(
            yaw_range[0], yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        target_quat_base = _quat_from_rpy(target_rpy_base)

        self.reach_use_left[env_ids] = target_pos_base[:, 1] > 0.0
        self.reach_arm_selector[env_ids, 0] = self.reach_use_left[env_ids].float()
        self.reach_arm_selector[env_ids, 1] = (~self.reach_use_left[env_ids]).float()

        self.reach_target_pos[env_ids] = \
            self.base_pos[env_ids] + quat_apply(self.base_quat[env_ids], target_pos_base)
        self.reach_target_pos_base[env_ids] = target_pos_base
        self.reach_target_rpy_base[env_ids] = target_rpy_base
        self.reach_target_quat[env_ids] = _quat_mul(self.base_quat[env_ids], target_quat_base)

    def _update_reach_target_base_frame(self):
        target_delta_world = self.reach_target_pos - self.base_pos
        self.reach_target_pos_base[:] = quat_rotate_inverse(self.base_quat, target_delta_world)

    def _active_end_effector_pos(self):
        left_pos = self.rigid_body_states[:, self.end_effector_indices[0], 0:3]
        right_pos = self.rigid_body_states[:, self.end_effector_indices[1], 0:3]
        return torch.where(self.reach_use_left.unsqueeze(1), left_pos, right_pos)

    def _active_end_effector_quat(self):
        left_quat = self.rigid_body_states[:, self.end_effector_indices[0], 3:7]
        right_quat = self.rigid_body_states[:, self.end_effector_indices[1], 3:7]
        return torch.where(self.reach_use_left.unsqueeze(1), left_quat, right_quat)

    def _upright_reach_gate(self):
        return (torch.abs(self.base_projected_gravity[:, 2]) > self.cfg.rewards.reach_upright_gate).float()

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        target_geom_left = gymutil.WireframeSphereGeometry(
            radius=0.045,
            num_lats=8,
            num_lons=8,
            pose=None,
            color=(0.1, 0.8, 1.0),
        )
        target_geom_right = gymutil.WireframeSphereGeometry(
            radius=0.045,
            num_lats=8,
            num_lons=8,
            pose=None,
            color=(1.0, 0.45, 0.1),
        )

        active_end_effector_pos = self._active_end_effector_pos()

        for env_id in range(self.num_envs):
            use_left = self.reach_use_left[env_id].item()
            target_pos = self.reach_target_pos[env_id].detach().cpu().numpy()
            wrist_pos = active_end_effector_pos[env_id].detach().cpu().numpy()

            sphere_pose = gymapi.Transform(
                gymapi.Vec3(target_pos[0], target_pos[1], target_pos[2]),
                r=None,
            )
            target_geom = target_geom_left if use_left else target_geom_right
            gymutil.draw_lines(target_geom, self.gym, self.viewer, self.env_handles[env_id], sphere_pose)

            line_vertices = np.array([wrist_pos, target_pos], dtype=np.float32)
            line_color = np.array([[0.1, 0.8, 1.0]], dtype=np.float32) \
                if use_left else np.array([[1.0, 0.45, 0.1]], dtype=np.float32)
            self.gym.add_lines(self.viewer, self.env_handles[env_id], 1, line_vertices, line_color)

    def _inactive_arm_action_error(self):
        left_error = torch.sum(torch.abs(self.actions[:, self.left_arm_action_indices]), dim=1)
        right_error = torch.sum(torch.abs(self.actions[:, self.right_arm_action_indices]), dim=1)
        return torch.where(self.reach_use_left, right_error, left_error)

    def _reward_reach_target_pos(self):
        target_error = torch.norm(self._active_end_effector_pos() - self.reach_target_pos, dim=1)
        return torch.exp(-target_error / self.cfg.rewards.reach_pos_tracking_sigma) * self._upright_reach_gate()

    def _reward_reach_target_success(self):
        target_error = torch.norm(self._active_end_effector_pos() - self.reach_target_pos, dim=1)
        return (target_error < self.cfg.rewards.reach_success_distance).float() * self._upright_reach_gate()

    def _reward_reach_target_stable(self):
        target_error = torch.norm(self._active_end_effector_pos() - self.reach_target_pos, dim=1)
        target_close = torch.exp(-target_error / self.cfg.rewards.reach_pos_tracking_sigma)
        lin_vel_error = torch.norm(self.base_lin_vel[:, 0:2], dim=1)
        ang_vel_error = torch.norm(self.base_ang_vel[:, 0:2], dim=1) + torch.abs(self.base_ang_vel[:, 2])
        base_stable = torch.exp(-lin_vel_error / self.cfg.rewards.reach_stable_lin_vel_sigma) \
            * torch.exp(-ang_vel_error / self.cfg.rewards.reach_stable_ang_vel_sigma)
        return target_close * base_stable * self._upright_reach_gate()

    def _reward_reach_target_orient(self):
        active_quat = self._active_end_effector_quat()
        quat_dot = torch.abs(torch.sum(active_quat * self.reach_target_quat, dim=1))
        quat_dot = torch.clamp(quat_dot, max=1.0)
        orient_error = 2.0 * torch.acos(quat_dot)
        return torch.exp(-orient_error / self.cfg.rewards.reach_orient_tracking_sigma)

    def _reward_inactive_arm_still(self):
        return self._inactive_arm_action_error()

    def _reward_base_still(self):
        return torch.norm(self.base_lin_vel[:, 0:2], dim=1) + torch.abs(self.base_ang_vel[:, 2])

    def _reward_main_body_dof_pos(self):
        main_body_offset = torch.abs(
            self.dof_pos[:, self.main_body_joint_indices]
            - self.default_dof_pos[:, self.main_body_joint_indices]
        )
        return torch.sum(main_body_offset, dim=1)

    def _reward_limits_dof_pos_without_ankle(self):
        related_indices = self.action_indices.tolist()
        related_indices = [i for i in related_indices if i not in self.ankle_indices]

        out_of_limits = -(self.dof_pos[:, related_indices]
                          - self.soft_dof_pos_limits[:, 0][related_indices]).clip(max=0.0)
        out_of_limits += (self.dof_pos[:, related_indices]
                          - self.soft_dof_pos_limits[:, 1][related_indices]).clip(min=0.0)

        error_limits_dof_pos = torch.sum(torch.abs(out_of_limits), dim=1)
        return 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_pos * error_limits_dof_pos)

    def _reward_limits_dof_vel_without_ankle(self):
        related_indices = self.action_indices.tolist()
        related_indices = [i for i in related_indices if i not in self.ankle_indices]

        out_of_limits = (torch.abs(self.dof_vel[:, related_indices])
                         - self.soft_dof_vel_limits[related_indices]).clip(min=0.0)

        error_limits_dof_vel = torch.sum(torch.abs(out_of_limits), dim=1)
        return 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_vel * error_limits_dof_vel)

    def _reward_limits_dof_tor(self):
        related_indices = self.action_indices.tolist()

        out_of_limits = (torch.abs(self.dof_tor[:, related_indices])
                         - self.soft_dof_tor_limits[related_indices]).clip(min=0.0)

        error_limits_dof_tor = torch.sum(torch.abs(out_of_limits), dim=1)
        return 1 - torch.exp(self.cfg.rewards.sigma_limits_dof_tor * error_limits_dof_tor)
