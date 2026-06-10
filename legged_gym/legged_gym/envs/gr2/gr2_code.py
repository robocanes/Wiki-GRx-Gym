from isaacgym.torch_utils import *

from legged_gym.envs.fftai.legged_robot_fftai_bipedal_code import LeggedRobotFFTAIBipedal
from legged_gym.envs.gr2.gr2_config import GR2Cfg


class GR2(LeggedRobotFFTAIBipedal):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg: GR2Cfg = cfg

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    def compute_observation_profile(self):

        obs_buf = torch.cat(
            (
                # command
                self.commands[:, 0:3] * self.commands_scale,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity * self.obs_scales.gravity,

                # dof related
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,

                # action related
                self.actions * self.obs_scales.action,
            ), dim=-1)

        pri_obs_buf = torch.cat(
            (
                obs_buf,

                # base related
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset * self.obs_scales.height_measurements,

                # foot related
                self.feet_contact,
                self.feet_height * self.obs_scales.height_measurements,
                self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,

                # terrain related
                self.surround_heights_offset * self.obs_scales.height_measurements,
            ), dim=-1)

        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def compute_obs_noise_scale_vec_profile(self):
        """
        Returns the noise scale vector for the observation vector.

        The noise scale vector is used to scale the noise vector to the same scale as the observation vector.

        Output:
        - noise_scale_vec: torch.Tensor
        """

        noise_vec = torch.zeros_like(self.obs_buf[0])

        # command
        start_index_of_commands = 0
        index_offset_of_commands = self.cfg.commands.num_commands
        noise_vec[start_index_of_commands + 0:
                  start_index_of_commands + self.cfg.commands.num_commands] = 0.  # commands x, y, yaw

        # base related
        start_index_of_base_related = start_index_of_commands + index_offset_of_commands
        index_offset_of_base_related = 6
        noise_vec[start_index_of_base_related + 0:
                  start_index_of_base_related + 3] = \
            self.noise_scales.ang_vel \
            * self.noise_level \
            * self.obs_scales.ang_vel  # base ang vel
        noise_vec[start_index_of_base_related + 3:
                  start_index_of_base_related + 6] = \
            self.noise_scales.gravity \
            * self.noise_level \
            * self.obs_scales.gravity  # base projected gravity

        # dof related
        start_index_of_dof_related = start_index_of_base_related + index_offset_of_base_related
        index_offset_of_dof_related = 2 * self.num_dofs
        noise_vec[start_index_of_dof_related + 0 * self.num_dofs:
                  start_index_of_dof_related + 1 * self.num_dofs] = \
            self.noise_scales.dof_pos \
            * self.noise_level \
            * self.obs_scales.dof_pos  # dof_pos_offset
        noise_vec[start_index_of_dof_related + 1 * self.num_dofs:
                  start_index_of_dof_related + 2 * self.num_dofs] = \
            self.noise_scales.dof_vel \
            * self.noise_level \
            * self.obs_scales.dof_vel  # dof_vel

        # action related
        start_index_of_action_related = start_index_of_dof_related + index_offset_of_dof_related
        index_offset_of_action_related = 1 * self.num_actions
        noise_vec[start_index_of_action_related + 0 * self.num_actions:
                  start_index_of_action_related + 1 * self.num_actions] = \
            self.noise_scales.action \
            * self.noise_level \
            * self.obs_scales.action  # actions

        return noise_vec

    # ----------------------------------------------

    def _resample_commands(self, env_ids=None, command_profile=None):
        super()._resample_commands(env_ids, command_profile)
        self.apply_dynamic_walk_yaw_curriculum(env_ids)

        self.update_gait_generator_pattern()

    def apply_dynamic_walk_yaw_curriculum(self, env_ids):
        if not getattr(self.cfg.commands, "yaw_spot_turn_curriculum", False):
            return
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, step=1, device=self.device)
        if len(env_ids) == 0:
            return

        spot_fraction = getattr(self.cfg.commands, "yaw_spot_turn_fraction", 0.0)
        if spot_fraction <= 0.0:
            return

        spot_mask = torch.rand(len(env_ids), device=self.device) < spot_fraction
        spot_env_ids = env_ids[spot_mask]
        if len(spot_env_ids) == 0:
            return

        min_abs_yaw = getattr(self.cfg.commands, "yaw_spot_turn_min_abs", 0.35)
        max_abs_yaw = getattr(self.cfg.commands, "yaw_spot_turn_max_abs", self.command_ranges["ang_vel_yaw"][1])
        positive_fraction = getattr(self.cfg.commands, "yaw_spot_turn_positive_fraction", 0.50)

        yaw_mag = torch_rand_float(
            min_abs_yaw,
            max_abs_yaw,
            (len(spot_env_ids), 1),
            device=self.device,
        ).squeeze(1)
        yaw_sign = torch.where(
            torch.rand(len(spot_env_ids), device=self.device) < positive_fraction,
            torch.ones(len(spot_env_ids), device=self.device),
            -torch.ones(len(spot_env_ids), device=self.device),
        )
        self.commands[spot_env_ids, 0] = 0.0
        self.commands[spot_env_ids, 1] = 0.0
        self.commands[spot_env_ids, 2] = yaw_mag * yaw_sign

        self.commands_base_lin_vel_x = self.commands[:, 0:1]
        self.commands_base_lin_vel_y = self.commands[:, 1:2]
        self.commands_base_ang_vel_yaw = self.commands[:, 2:3]

    def set_commands(self, env_ids, commands):
        """
        Sets the commands for the specified environments.

        NOTE: should not be called in the training process!!!
        """
        self.commands[env_ids] = commands

        self._command_refinement(env_ids)
        self.update_gait_generator_pattern()

    def update_flags_of_stand_command(self):
        # situation 1: the norm of the command x, y lin_vel is less than 0.10
        flags_of_stand_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) <= 0.10

        # situation 2: the value of the command yaw ang_vel is less than 0.10
        flags_of_stand_command_s2 = torch.abs(self.commands[:, 2]) <= 0.10

        # situation 1 and situation 2
        flags_of_stand_command = flags_of_stand_command_s1 * flags_of_stand_command_s2
        flags_of_stand_command = flags_of_stand_command.bool()

        return flags_of_stand_command

    def update_flags_of_walk_command(self):
        # situation 1: the norm of the command x, y lin_vel is greater than 0.10
        flags_of_walk_command_s1 = torch.norm(self.commands[:, 0:2], dim=1) > 0.10

        # situation 2: the value of the command yaw ang_vel is greater than 0.10
        flags_of_walk_command_s2 = torch.abs(self.commands[:, 2]) > 0.10

        # situation 1 and situation 2
        flags_of_walk_command = flags_of_walk_command_s1 + flags_of_walk_command_s2
        flags_of_walk_command = flags_of_walk_command.bool()

        return flags_of_walk_command

    def update_gait_generator_pattern(self):
        env_ids_of_off_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_stand_command = torch.Tensor([]).int().to(self.device)
        env_ids_of_walk_command = torch.Tensor([]).int().to(self.device)

        if "stand" in self.cfg.commands.gait_patterns:
            env_ids_of_stand_command = torch.where(self.update_flags_of_stand_command())[0]

        if "walk" in self.cfg.commands.gait_patterns:
            env_ids_of_walk_command = torch.where(self.update_flags_of_walk_command())[0]

        # update the env_ids of different gait patterns
        self.env_ids_of_off_command = env_ids_of_off_command
        self.env_ids_of_stand_command = env_ids_of_stand_command
        self.env_ids_of_walk_command = env_ids_of_walk_command

    def _shoulder_sideways_indices(self):
        return self.shoulder_roll_indices + self.shoulder_yaw_indices

    def _shoulder_pitch_indices(self):
        return self.shoulder_pitch_indices

    def _reward_shoulder_pitch_pos(self):
        shoulder_pitch_indices = self._shoulder_pitch_indices()
        if len(shoulder_pitch_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_pitch_offset = torch.abs(
            self.dof_pos[:, shoulder_pitch_indices]
            - self.default_dof_pos[:, shoulder_pitch_indices]
        )
        return torch.sum(shoulder_pitch_offset, dim=1)

    def _reward_shoulder_pitch_vel(self):
        shoulder_pitch_indices = self._shoulder_pitch_indices()
        if len(shoulder_pitch_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_pitch_vel = torch.abs(self.dof_vel[:, shoulder_pitch_indices])
        return torch.sum(shoulder_pitch_vel, dim=1)

    def _reward_shoulder_pitch_swing(self):
        shoulder_pitch_indices = self._shoulder_pitch_indices()
        if len(shoulder_pitch_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_pitch_offset = (
            self.dof_pos[:, shoulder_pitch_indices]
            - self.default_dof_pos[:, shoulder_pitch_indices]
        )
        left_pitch = shoulder_pitch_offset[:, 0]
        right_pitch = shoulder_pitch_offset[:, 1]
        anti_phase_swing = torch.clamp(torch.abs(left_pitch - right_pitch), max=0.70)

        walk_mask = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "env_ids_of_walk_command"):
            walk_mask[self.env_ids_of_walk_command] = 1.0
        else:
            walk_mask[:] = 1.0

        return anti_phase_swing * walk_mask

    def _reward_shoulder_sideways_pos(self):
        shoulder_sideways_indices = self._shoulder_sideways_indices()
        if len(shoulder_sideways_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_sideways_offset = torch.abs(
            self.dof_pos[:, shoulder_sideways_indices]
            - self.default_dof_pos[:, shoulder_sideways_indices]
        )
        return torch.sum(shoulder_sideways_offset, dim=1)

    def _reward_shoulder_sideways_vel(self):
        shoulder_sideways_indices = self._shoulder_sideways_indices()
        if len(shoulder_sideways_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_sideways_vel = torch.abs(self.dof_vel[:, shoulder_sideways_indices])
        return torch.sum(shoulder_sideways_vel, dim=1)

    def _walk_mask(self):
        mask = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "env_ids_of_walk_command"):
            mask[self.env_ids_of_walk_command] = 1.0
        else:
            mask[:] = 1.0
        return mask

    def _feet_positions_in_base_frame(self):
        left_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 0, 0:3]
        right_foot_pos_in_world_frame = self.rigid_body_states[:, self.feet_indices][:, 1, 0:3]

        left_foot_pos_to_base_in_world_frame = left_foot_pos_in_world_frame - self.root_states[:, 0:3]
        right_foot_pos_to_base_in_world_frame = right_foot_pos_in_world_frame - self.root_states[:, 0:3]

        left_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], left_foot_pos_to_base_in_world_frame)
        right_foot_pos = quat_rotate_inverse(self.root_states[:, 3:7], right_foot_pos_to_base_in_world_frame)
        return left_foot_pos, right_foot_pos

    def _reward_feet_swing_clearance(self):
        walk_mask = self._walk_mask()
        swing_mask = (~self.feet_contact).float()
        clearance_error = torch.clip(self.cfg.rewards.feet_swing_clearance_target - self.feet_height, min=0.0)
        return torch.sum(clearance_error * swing_mask, dim=1) * walk_mask

    def _reward_feet_step_length_symmetry(self):
        walk_mask = self._walk_mask()
        contact_event = torch.any(self.feet_contact_trig, dim=1).float()
        left_foot_pos, right_foot_pos = self._feet_positions_in_base_frame()
        step_length_error = torch.abs(torch.abs(left_foot_pos[:, 0]) - torch.abs(right_foot_pos[:, 0]))
        return step_length_error * contact_event * walk_mask

    def _reward_feet_air_time_symmetry(self):
        walk_mask = self._walk_mask()
        contact_event = torch.any(self.feet_contact_trig, dim=1).float()
        air_time_error = torch.abs(self.feet_air_time_last[:, 0] - self.feet_air_time_last[:, 1])
        air_time_error = air_time_error / max(self.cfg.rewards.feet_air_time_target, 1.0e-6)
        return air_time_error * contact_event * walk_mask

    def _ankle_action_indices(self):
        return self.ankle_pitch_indices + self.ankle_roll_indices

    def _ankle_pitch_action_indices(self):
        return self.ankle_pitch_indices

    def _reward_ankle_action_abs(self):
        ankle_indices = self._ankle_action_indices()
        if len(ankle_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        return torch.sum(torch.abs(self.actions[:, ankle_indices]), dim=1) * self._walk_mask()

    def _reward_ankle_action_rate(self):
        ankle_indices = self._ankle_action_indices()
        if len(ankle_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        ankle_action_delta = self.actions[:, ankle_indices] - self.last_actions[:, ankle_indices]
        return torch.sum(torch.square(ankle_action_delta), dim=1) * self._walk_mask()

    def _reward_ankle_pitch_pos_offset(self):
        ankle_pitch_indices = self._ankle_pitch_action_indices()
        if len(ankle_pitch_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        ankle_pitch_offset = torch.abs(
            self.dof_pos[:, ankle_pitch_indices]
            - self.default_dof_pos[:, ankle_pitch_indices]
        )
        return torch.sum(ankle_pitch_offset, dim=1) * self._walk_mask()

    def _reward_cmd_diff_base_ang_vel_yaw_positive(self):
        min_command = getattr(self.cfg.rewards, "positive_yaw_reward_min_command", 0.15)
        positive_yaw_mask = (self.commands_base_ang_vel_yaw[:, 0] > min_command).float()
        error_yaw_vel = torch.abs(self.commands_base_ang_vel_yaw - self.base_ang_vel[:, 2:3])
        error_yaw_vel = torch.sum(error_yaw_vel, dim=1)
        reward_yaw_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_ang_vel_yaw * error_yaw_vel)
        return reward_yaw_vel * positive_yaw_mask

    def _reward_cmd_diff_base_ang_vel_yaw_spot(self):
        min_command = getattr(self.cfg.rewards, "spot_yaw_reward_min_command", 0.15)
        max_linear_command = getattr(self.cfg.rewards, "spot_yaw_reward_max_linear_command", 0.10)
        spot_yaw_mask = (
            (torch.abs(self.commands_base_ang_vel_yaw[:, 0]) > min_command)
            & (torch.norm(self.commands[:, 0:2], dim=1) <= max_linear_command)
        ).float()
        error_yaw_vel = torch.abs(self.commands_base_ang_vel_yaw - self.base_ang_vel[:, 2:3])
        error_yaw_vel = torch.sum(error_yaw_vel, dim=1)
        reward_yaw_vel = torch.exp(self.cfg.rewards.sigma_cmd_diff_base_ang_vel_yaw * error_yaw_vel)
        return reward_yaw_vel * spot_yaw_mask

    # ==========================================================================================================================
    # Reward functions


class GR2DynamicWalk12(GR2):
    """Vendor-style lower-body walk: 12 leg actions, 49 actor observations."""

    def _gait_phase_obs(self):
        gait_cycle_period = getattr(self.cfg.rewards, "gait_cycle_period", 0.8)
        phase = self.episode_length_buf.float() * self.dt / gait_cycle_period * 2.0 * torch.pi
        return torch.stack((torch.sin(phase), torch.cos(phase)), dim=1)

    def compute_observation_profile(self):
        gait_phase = self._gait_phase_obs()

        obs_buf = torch.cat(
            (
                # command
                self.commands[:, 0:3] * self.commands_scale,

                # base related
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.base_projected_gravity * self.obs_scales.gravity,

                # dof related. The waist is observed but not controlled.
                self.dof_pos_offset * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,

                # action related
                self.actions * self.obs_scales.action,

                # CPG-compatible phase cue
                gait_phase,
            ), dim=-1)

        pri_obs_buf = torch.cat(
            (
                obs_buf,

                # base related
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_heights_offset * self.obs_scales.height_measurements,

                # foot related
                self.feet_contact,
                self.feet_height * self.obs_scales.height_measurements,
                self.avg_feet_speed_xyz[:, 0, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 0:1] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 0, 1:2] * self.obs_scales.lin_vel,
                self.avg_feet_speed_xyz[:, 1, 1:2] * self.obs_scales.lin_vel,

                # terrain related
                self.surround_heights_offset * self.obs_scales.height_measurements,
            ), dim=-1)

        self.obs_buf = obs_buf
        self.pri_obs_buf = pri_obs_buf

    def compute_obs_noise_scale_vec_profile(self):
        return super().compute_obs_noise_scale_vec_profile()

    def _reward_double_support_stamping(self):
        walk_mask = self._walk_mask()
        double_support = (self.feet_contact[:, 0] & self.feet_contact[:, 1]).float()
        vertical_force = torch.sum(torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        robot_weight = getattr(self.cfg.rewards, "robot_mass", 66.73034) * 9.81
        normalized_force = vertical_force / max(robot_weight, 1.0e-6)
        return double_support * torch.clamp(normalized_force - 1.0, min=0.0) * walk_mask

    def _reward_ankle_pitch_motion(self):
        ankle_pitch_indices = self._ankle_pitch_action_indices()
        if len(ankle_pitch_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        ankle_pitch_vel = torch.abs(self.dof_vel[:, ankle_pitch_indices])
        useful_motion = torch.clamp(torch.sum(ankle_pitch_vel, dim=1), max=4.0)
        return useful_motion * self._walk_mask()
