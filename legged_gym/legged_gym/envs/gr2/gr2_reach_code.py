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


def _quat_apply_np(q, v):
    q_xyz = q[:3]
    q_w = q[3]
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (q_w * uv + uuv)


def _transform_point_np(pos, quat, point):
    return pos + _quat_apply_np(quat, point)


class GR2Reach(GR2):
    def _init_buffers_others(self):
        super()._init_buffers_others()

        self.reach_target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.reach_target_pos_base = torch.zeros_like(self.reach_target_pos)
        self.reach_target_rpy_base = torch.zeros_like(self.reach_target_pos)
        self.reach_target_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.reach_target_origin_pos = torch.zeros_like(self.reach_target_pos)
        self.reach_target_origin_quat = torch.zeros_like(self.reach_target_quat)
        self.reach_use_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reach_arm_selector = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.camera_indices = torch.tensor(
            [i for i, name in enumerate(self.body_names) if "camera_link" in name],
            dtype=torch.long,
            device=self.device,
        )
        if len(self.camera_indices) == 0:
            self.camera_indices = self.head_indices

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
        edge_sample_prob = float(getattr(self.cfg.reach, "target_edge_sample_prob", 0.0))
        edge_z_band_fraction = float(getattr(self.cfg.reach, "target_edge_z_band_fraction", 0.25))
        edge_y_band_fraction = float(getattr(self.cfg.reach, "target_edge_y_band_fraction", 0.30))
        edge_x_band_fraction = float(getattr(self.cfg.reach, "target_edge_x_band_fraction", 0.30))
        edge_low_z_weight = float(getattr(self.cfg.reach, "target_edge_low_z_weight", 0.35))
        edge_high_z_weight = float(getattr(self.cfg.reach, "target_edge_high_z_weight", 0.35))
        edge_outer_y_weight = float(getattr(self.cfg.reach, "target_edge_outer_y_weight", 0.20))
        edge_forward_x_weight = float(getattr(self.cfg.reach, "target_edge_forward_x_weight", 0.10))

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

        if edge_sample_prob > 0.0:
            num_targets = len(env_ids)
            edge_mask = torch.rand(num_targets, device=self.device) < edge_sample_prob
            edge_count = int(edge_mask.sum().item())
            if edge_count > 0:
                z_span = z_range[1] - z_range[0]
                y_span = y_abs_range[1] - y_abs_range[0]
                x_span = x_range[1] - x_range[0]
                z_band = max(z_span * edge_z_band_fraction, 1.0e-6)
                y_band = max(y_span * edge_y_band_fraction, 1.0e-6)
                x_band = max(x_span * edge_x_band_fraction, 1.0e-6)

                mode_weights = torch.tensor(
                    [
                        max(edge_low_z_weight, 0.0),
                        max(edge_high_z_weight, 0.0),
                        max(edge_outer_y_weight, 0.0),
                        max(edge_forward_x_weight, 0.0),
                    ],
                    dtype=torch.float,
                    device=self.device,
                )
                mode_weights = mode_weights / torch.clamp(torch.sum(mode_weights), min=1.0e-6)
                mode_ids = torch.multinomial(mode_weights, edge_count, replacement=True)
                edge_indices = torch.nonzero(edge_mask, as_tuple=False).squeeze(1)

                low_z_mask = mode_ids == 0
                if torch.any(low_z_mask):
                    indices = edge_indices[low_z_mask]
                    target_pos_base[indices, 2] = torch_rand_float(
                        z_range[0], z_range[0] + z_band, (len(indices), 1), device=self.device
                    ).squeeze(1)

                high_z_mask = mode_ids == 1
                if torch.any(high_z_mask):
                    indices = edge_indices[high_z_mask]
                    target_pos_base[indices, 2] = torch_rand_float(
                        z_range[1] - z_band, z_range[1], (len(indices), 1), device=self.device
                    ).squeeze(1)

                outer_y_mask = mode_ids == 2
                if torch.any(outer_y_mask):
                    indices = edge_indices[outer_y_mask]
                    target_y_abs = torch_rand_float(
                        y_abs_range[1] - y_band, y_abs_range[1], (len(indices), 1), device=self.device
                    ).squeeze(1)
                    target_pos_base[indices, 1] = target_y_abs * target_side[indices]

                forward_x_mask = mode_ids == 3
                if torch.any(forward_x_mask):
                    indices = edge_indices[forward_x_mask]
                    target_pos_base[indices, 0] = torch_rand_float(
                        x_range[1] - x_band, x_range[1], (len(indices), 1), device=self.device
                    ).squeeze(1)

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
        self.reach_target_origin_pos[env_ids] = self.base_pos[env_ids]
        self.reach_target_origin_quat[env_ids] = self.base_quat[env_ids]

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

    def _active_end_effector_vel(self):
        left_vel = self.rigid_body_states[:, self.end_effector_indices[0], 7:10]
        right_vel = self.rigid_body_states[:, self.end_effector_indices[1], 7:10]
        return torch.where(self.reach_use_left.unsqueeze(1), left_vel, right_vel)

    def _upright_reach_gate(self):
        return (torch.abs(self.base_projected_gravity[:, 2]) > self.cfg.rewards.reach_upright_gate).float()

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        active_end_effector_pos = self._active_end_effector_pos()
        axis_length = 0.11
        tick_length = 0.025
        axis_basis = np.eye(3, dtype=np.float32)
        axis_colors = np.array(
            [
                [1.0, 0.1, 0.1],
                [0.1, 0.85, 0.1],
                [0.15, 0.35, 1.0],
            ],
            dtype=np.float32,
        )
        draw_target_volume_box = getattr(self.cfg.reach, "draw_target_volume_box", True)
        if draw_target_volume_box:
            box_color = np.array([[0.9, 0.9, 0.9]], dtype=np.float32)
            x_min, x_max = self.cfg.reach.target_x_range
            y_max = self.cfg.reach.target_y_abs_range[1]
            z_min, z_max = self.cfg.reach.target_z_range
            box_corners_base = np.array(
                [
                    [x_min, -y_max, z_min],
                    [x_max, -y_max, z_min],
                    [x_max, y_max, z_min],
                    [x_min, y_max, z_min],
                    [x_min, -y_max, z_max],
                    [x_max, -y_max, z_max],
                    [x_max, y_max, z_max],
                    [x_min, y_max, z_max],
                ],
                dtype=np.float32,
            )
            box_edges = (
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            )

        for env_id in range(self.num_envs):
            use_left = self.reach_use_left[env_id].item()
            target_origin_pos = self.reach_target_origin_pos[env_id].detach().cpu().numpy()
            target_origin_quat = self.reach_target_origin_quat[env_id].detach().cpu().numpy()
            target_pos = self.reach_target_pos[env_id].detach().cpu().numpy()
            target_quat = self.reach_target_quat[env_id].detach().cpu().numpy()
            wrist_pos = active_end_effector_pos[env_id].detach().cpu().numpy()

            if draw_target_volume_box:
                box_corners = np.array(
                    [_transform_point_np(target_origin_pos, target_origin_quat, corner) for corner in box_corners_base],
                    dtype=np.float32,
                )
                box_vertices = np.array(
                    [box_corners[index] for edge in box_edges for index in edge],
                    dtype=np.float32,
                )
                self.gym.add_lines(
                    self.viewer,
                    self.env_handles[env_id],
                    len(box_edges),
                    box_vertices,
                    np.repeat(box_color, len(box_edges), axis=0),
                )

            marker_vertices = []
            marker_colors = []
            for axis_id, local_axis in enumerate(axis_basis):
                axis = _quat_apply_np(target_quat, local_axis)
                end_pos = target_pos + axis * axis_length
                marker_vertices.extend([target_pos, end_pos])
                marker_colors.append(axis_colors[axis_id])

                tick_axis = _quat_apply_np(target_quat, axis_basis[(axis_id + 1) % 3])
                marker_vertices.extend([end_pos, end_pos - axis * tick_length + tick_axis * tick_length])
                marker_vertices.extend([end_pos, end_pos - axis * tick_length - tick_axis * tick_length])
                marker_colors.extend([axis_colors[axis_id], axis_colors[axis_id]])

            marker_vertices = np.array(marker_vertices, dtype=np.float32)
            marker_colors = np.array(marker_colors, dtype=np.float32)
            self.gym.add_lines(self.viewer, self.env_handles[env_id], len(marker_colors), marker_vertices, marker_colors)

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

    def _reward_active_end_effector_still(self):
        target_error = torch.norm(self._active_end_effector_pos() - self.reach_target_pos, dim=1)
        target_close = torch.exp(-target_error / self.cfg.rewards.reach_pos_tracking_sigma)
        end_effector_speed = torch.norm(self._active_end_effector_vel(), dim=1)
        speed_penalty = 1.0 - torch.exp(-end_effector_speed / self.cfg.rewards.reach_ee_vel_sigma)
        return speed_penalty * target_close * self._upright_reach_gate()

    def _reward_reach_target_orient(self):
        active_quat = self._active_end_effector_quat()
        quat_dot = torch.abs(torch.sum(active_quat * self.reach_target_quat, dim=1))
        quat_dot = torch.clamp(quat_dot, max=1.0)
        orient_error = 2.0 * torch.acos(quat_dot)
        return torch.exp(-orient_error / self.cfg.rewards.reach_orient_tracking_sigma)

    def _reward_head_look_at_target(self):
        if len(self.camera_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        camera_pos = self.rigid_body_states[:, self.camera_indices[0], 0:3]
        camera_quat = self.rigid_body_states[:, self.camera_indices[0], 3:7]
        camera_forward = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        camera_forward[:, 0] = 1.0
        camera_forward = quat_apply(camera_quat, camera_forward)

        target_direction = self.reach_target_pos - camera_pos
        target_direction = target_direction / torch.clamp(torch.norm(target_direction, dim=1, keepdim=True), min=1e-6)
        look_alignment = torch.sum(camera_forward * target_direction, dim=1)
        return torch.clamp(look_alignment, min=0.0) * self._upright_reach_gate()

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


class GR2DynamicWalkReachPretrain(GR2Reach):
    """Reach-shaped full-body policy pretraining with dynamic-walk commands."""

    def compute_observation_variables(self):
        GR2.compute_observation_variables(self)

    def compute_observation_profile(self):
        command_context = torch.cat(
            (
                self.commands[:, 0:3] * self.commands_scale,
                torch.zeros(self.num_envs, 5, dtype=torch.float, device=self.device),
            ), dim=-1)

        obs_buf = torch.cat(
            (
                command_context,

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

    def _resample_commands(self, env_ids=None, command_profile=None):
        GR2._resample_commands(self, env_ids, command_profile)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) > 0:
            self._sample_reach_targets(env_ids)
