import torch

from legged_gym.envs.gr2.gr2_config_reach_pickup_high import (
    GR2ReachPickupHighCfg,
    GR2ReachPickupHighCfgPPO,
)
from legged_gym.envs.gr2.gr2_reach_code import GR2Reach


class GR2ReachPlantedPickup(GR2Reach):
    """Reach task variant that makes floor-pickup a planted squat, not a shuffle."""

    def _init_buffers_others(self):
        super()._init_buffers_others()

        action_dof_names = [self.dof_names[i] for i in self.action_indices.tolist()]
        self.knee_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if "knee_pitch" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.hip_pitch_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if "hip_pitch" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.ankle_pitch_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if "ankle_pitch" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.yaw_action_indices = torch.tensor(
            [
                i for i, name in enumerate(action_dof_names)
                if "hip_yaw" in name or "waist_yaw" in name
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.knee_dof_indices = torch.tensor(
            [
                int(self.action_indices[i].item())
                for i in self.knee_action_indices.tolist()
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.hip_pitch_dof_indices = torch.tensor(
            [
                int(self.action_indices[i].item())
                for i in self.hip_pitch_action_indices.tolist()
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.ankle_pitch_dof_indices = torch.tensor(
            [
                int(self.action_indices[i].item())
                for i in self.ankle_pitch_action_indices.tolist()
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.yaw_dof_indices = torch.tensor(
            [
                int(self.action_indices[i].item())
                for i in self.yaw_action_indices.tolist()
            ],
            dtype=torch.long,
            device=self.device,
        )
        left_shoulder_candidates = [
            i for i, name in enumerate(self.body_names)
            if name.startswith("left_upper_arm_pitch")
        ]
        right_shoulder_candidates = [
            i for i, name in enumerate(self.body_names)
            if name.startswith("right_upper_arm_pitch")
        ]
        if left_shoulder_candidates and right_shoulder_candidates:
            self.reach_shoulder_indices = torch.tensor(
                [left_shoulder_candidates[0], right_shoulder_candidates[0]],
                dtype=torch.long,
                device=self.device,
            )
        else:
            self.reach_shoulder_indices = torch.tensor([], dtype=torch.long, device=self.device)

        self.episode_start_base_pos = self.base_pos.clone()
        self.episode_start_feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3].clone()
        self.episode_start_feet_width = torch.norm(
            self.episode_start_feet_pos[:, 0, 0:2] - self.episode_start_feet_pos[:, 1, 0:2],
            dim=1,
        )
        if len(self.reach_shoulder_indices) == 2:
            self.episode_start_shoulder_pos = self.rigid_body_states[
                :, self.reach_shoulder_indices, 0:3
            ].clone()
        else:
            self.episode_start_shoulder_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)

    def _reset_others(self, env_ids):
        super()._reset_others(env_ids)
        self.episode_start_base_pos[env_ids] = self.base_pos[env_ids]
        self.episode_start_feet_pos[env_ids] = self.rigid_body_states[
            env_ids[:, None], self.feet_indices, 0:3
        ]
        self.episode_start_feet_width[env_ids] = torch.norm(
            self.episode_start_feet_pos[env_ids, 0, 0:2] - self.episode_start_feet_pos[env_ids, 1, 0:2],
            dim=1,
        )
        if len(self.reach_shoulder_indices) == 2:
            self.episode_start_shoulder_pos[env_ids] = self.rigid_body_states[
                env_ids[:, None], self.reach_shoulder_indices, 0:3
            ]

    def _low_target_gate(self):
        z = self.reach_target_pos_base[:, 2]
        low_start = float(self.cfg.rewards.pickup_low_target_start_z)
        low_full = float(self.cfg.rewards.pickup_low_target_full_z)
        return torch.clamp((low_start - z) / max(low_start - low_full, 1.0e-6), 0.0, 1.0)

    def _reward_base_xy_displacement(self):
        base_delta_xy = self.base_pos[:, 0:2] - self.episode_start_base_pos[:, 0:2]
        deadband = float(self.cfg.rewards.pickup_base_xy_deadband)
        return torch.clamp(torch.norm(base_delta_xy, dim=1) - deadband, min=0.0)

    def _current_feet_xy_width(self):
        feet_xy = self.rigid_body_states[:, self.feet_indices, 0:2]
        return torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=1)

    def _reward_feet_xy_displacement(self):
        feet_delta_xy = (
            self.rigid_body_states[:, self.feet_indices, 0:2]
            - self.episode_start_feet_pos[:, :, 0:2]
        )
        deadband = float(self.cfg.rewards.pickup_feet_xy_deadband)
        foot_norm = torch.norm(feet_delta_xy, dim=2)
        raw_foot_disp = torch.sum(torch.clamp(foot_norm - deadband, min=0.0), dim=1)

        gate = self._low_target_gate()
        feet_mid_delta = torch.mean(feet_delta_xy, dim=1)
        midpoint_disp = torch.clamp(torch.norm(feet_mid_delta, dim=1) - deadband, min=0.0)
        spread_allowance = float(self.cfg.rewards.pickup_feet_spread_allowance)
        excess_individual = torch.sum(
            torch.clamp(foot_norm - deadband - spread_allowance, min=0.0),
            dim=1,
        )
        planted_squat_disp = 2.0 * midpoint_disp + 0.35 * excess_individual
        return raw_foot_disp * (1.0 - gate) + planted_squat_disp * gate

    def _reward_low_target_stance_width(self):
        width_gain = torch.clamp(
            self._current_feet_xy_width() - self.episode_start_feet_width,
            min=0.0,
        )
        target = float(self.cfg.rewards.pickup_feet_width_increase_target)
        return torch.clamp(width_gain / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_abs_stance_width(self):
        width = self._current_feet_xy_width()
        max_width = float(self.cfg.rewards.pickup_feet_width_max)
        return torch.clamp(width - max_width, min=0.0) * self._low_target_gate()

    def _reward_low_target_feet_midpoint_drift(self):
        feet_xy = self.rigid_body_states[:, self.feet_indices, 0:2]
        feet_midpoint = torch.mean(feet_xy, dim=1)
        start_midpoint = torch.mean(self.episode_start_feet_pos[:, :, 0:2], dim=1)
        drift = torch.norm(feet_midpoint - start_midpoint, dim=1)
        deadband = float(self.cfg.rewards.pickup_feet_midpoint_deadband)
        return torch.clamp(drift - deadband, min=0.0) * self._low_target_gate()

    def _reward_feet_speed_xy(self):
        return torch.sum(torch.norm(self.rigid_body_states[:, self.feet_indices, 7:9], dim=2), dim=1)

    def _reward_low_target_knee_flexion(self):
        if len(self.knee_dof_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        knee_offset = torch.clamp(
            self.dof_pos[:, self.knee_dof_indices]
            - self.default_dof_pos[:, self.knee_dof_indices],
            min=0.0,
        )
        knee_flexion = torch.mean(knee_offset, dim=1)
        target = float(self.cfg.rewards.pickup_knee_flexion_target)
        return torch.clamp(knee_flexion / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_excess_knee_flexion(self):
        if len(self.knee_dof_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        knee_offset = torch.clamp(
            self.dof_pos[:, self.knee_dof_indices]
            - self.default_dof_pos[:, self.knee_dof_indices],
            min=0.0,
        )
        knee_flexion = torch.mean(knee_offset, dim=1)
        target = float(self.cfg.rewards.pickup_knee_flexion_target)
        allowance = float(self.cfg.rewards.pickup_knee_flexion_allowance)
        return torch.clamp(knee_flexion - target - allowance, min=0.0) * self._low_target_gate()

    def _reward_low_target_hip_flexion(self):
        if len(self.hip_pitch_dof_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        hip_offset = torch.abs(
            self.dof_pos[:, self.hip_pitch_dof_indices]
            - self.default_dof_pos[:, self.hip_pitch_dof_indices]
        )
        hip_flexion = torch.mean(hip_offset, dim=1)
        target = float(self.cfg.rewards.pickup_hip_flexion_target)
        return torch.clamp(hip_flexion / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_base_height_drop(self):
        base_drop = torch.clamp(self.episode_start_base_pos[:, 2] - self.base_pos[:, 2], min=0.0)
        target = float(self.cfg.rewards.pickup_base_height_drop_target)
        return torch.clamp(base_drop / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_excess_base_height_drop(self):
        base_drop = torch.clamp(self.episode_start_base_pos[:, 2] - self.base_pos[:, 2], min=0.0)
        target = float(self.cfg.rewards.pickup_base_height_drop_target)
        allowance = float(self.cfg.rewards.pickup_base_height_drop_allowance)
        return torch.clamp(base_drop - target - allowance, min=0.0) * self._low_target_gate()

    def _reward_low_target_lower_body_ground_clearance(self):
        lower_parts = []
        for name in ("thigh_indices", "shank_indices"):
            indices = getattr(self, name, None)
            if indices is not None and len(indices) > 0:
                lower_parts.append(indices)

        if len(lower_parts) == 0 or len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        lower_indices = torch.cat(lower_parts)
        lower_z = self.rigid_body_states[:, lower_indices, 2]
        foot_z = torch.mean(self.rigid_body_states[:, self.feet_indices, 2], dim=1)
        min_clearance = torch.min(lower_z, dim=1).values - foot_z
        target = float(self.cfg.rewards.pickup_lower_body_ground_clearance_min)
        return torch.clamp((target - min_clearance) / max(target, 1.0e-6), min=0.0) * self._low_target_gate()

    def _reward_low_target_lower_body_contact(self):
        lower_parts = []
        for name in ("thigh_indices", "shank_indices"):
            indices = getattr(self, name, None)
            if indices is not None and len(indices) > 0:
                lower_parts.append(indices)

        if len(lower_parts) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        lower_indices = torch.cat(lower_parts)
        threshold = float(self.cfg.rewards.pickup_lower_body_contact_force_threshold)
        contacts = torch.norm(self.contact_forces[:, lower_indices, :], dim=-1) > threshold
        return torch.sum(contacts.float(), dim=1) * self._low_target_gate()

    def _reward_low_target_ankle_pitch(self):
        if len(self.ankle_pitch_dof_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        ankle_offset = torch.abs(
            self.dof_pos[:, self.ankle_pitch_dof_indices]
            - self.default_dof_pos[:, self.ankle_pitch_dof_indices]
        )
        ankle_pitch = torch.mean(ankle_offset, dim=1)
        target = float(self.cfg.rewards.pickup_ankle_pitch_target)
        return torch.clamp(ankle_pitch / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_yaw_twist(self):
        if len(self.yaw_dof_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        yaw_offset = torch.abs(
            self.dof_pos[:, self.yaw_dof_indices]
            - self.default_dof_pos[:, self.yaw_dof_indices]
        )
        yaw_twist = torch.mean(yaw_offset, dim=1) + 0.20 * torch.abs(self.base_ang_vel[:, 2])
        deadband = float(self.cfg.rewards.pickup_yaw_twist_deadband)
        return torch.clamp(yaw_twist - deadband, min=0.0) * self._low_target_gate()

    def _reward_low_target_squat_coherence(self):
        if (
            len(self.knee_dof_indices) == 0
            or len(self.hip_pitch_dof_indices) == 0
            or len(self.ankle_pitch_dof_indices) == 0
        ):
            return torch.zeros(self.num_envs, device=self.device)

        knee_offset = torch.clamp(
            self.dof_pos[:, self.knee_dof_indices]
            - self.default_dof_pos[:, self.knee_dof_indices],
            min=0.0,
        )
        hip_offset = torch.abs(
            self.dof_pos[:, self.hip_pitch_dof_indices]
            - self.default_dof_pos[:, self.hip_pitch_dof_indices]
        )
        ankle_offset = torch.abs(
            self.dof_pos[:, self.ankle_pitch_dof_indices]
            - self.default_dof_pos[:, self.ankle_pitch_dof_indices]
        )
        base_drop = torch.clamp(self.episode_start_base_pos[:, 2] - self.base_pos[:, 2], min=0.0)

        knee_score = torch.clamp(
            torch.mean(knee_offset, dim=1) / max(float(self.cfg.rewards.pickup_knee_flexion_target), 1.0e-6),
            0.0,
            1.0,
        )
        hip_score = torch.clamp(
            torch.mean(hip_offset, dim=1) / max(float(self.cfg.rewards.pickup_hip_flexion_target), 1.0e-6),
            0.0,
            1.0,
        )
        ankle_score = torch.clamp(
            torch.mean(ankle_offset, dim=1) / max(float(self.cfg.rewards.pickup_ankle_pitch_target), 1.0e-6),
            0.0,
            1.0,
        )
        drop_score = torch.clamp(
            base_drop / max(float(self.cfg.rewards.pickup_base_height_drop_target), 1.0e-6),
            0.0,
            1.0,
        )
        return torch.minimum(
            torch.minimum(knee_score, hip_score),
            torch.minimum(ankle_score, drop_score),
        ) * self._low_target_gate()

    def _reward_low_target_leg_asymmetry(self):
        if len(self.knee_dof_indices) < 2 or len(self.ankle_pitch_dof_indices) < 2:
            return torch.zeros(self.num_envs, device=self.device)

        knee_offset = torch.clamp(
            self.dof_pos[:, self.knee_dof_indices]
            - self.default_dof_pos[:, self.knee_dof_indices],
            min=0.0,
        )
        ankle_offset = torch.abs(
            self.dof_pos[:, self.ankle_pitch_dof_indices]
            - self.default_dof_pos[:, self.ankle_pitch_dof_indices]
        )
        knee_asymmetry = torch.abs(knee_offset[:, 0] - knee_offset[:, 1])
        ankle_asymmetry = torch.abs(ankle_offset[:, 0] - ankle_offset[:, 1])
        return (knee_asymmetry + 0.65 * ankle_asymmetry) * self._low_target_gate()

    def _reward_low_target_active_shoulder_drop(self):
        if len(self.reach_shoulder_indices) != 2:
            return torch.zeros(self.num_envs, device=self.device)

        shoulder_pos = self.rigid_body_states[:, self.reach_shoulder_indices, 0:3]
        active_index = torch.where(
            self.reach_use_left,
            torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            torch.ones(self.num_envs, dtype=torch.long, device=self.device),
        )
        active_start_z = self.episode_start_shoulder_pos[
            torch.arange(self.num_envs, device=self.device), active_index, 2
        ]
        active_z = shoulder_pos[torch.arange(self.num_envs, device=self.device), active_index, 2]
        shoulder_drop = torch.clamp(active_start_z - active_z, min=0.0)
        target = float(self.cfg.rewards.pickup_shoulder_drop_target)
        return torch.clamp(shoulder_drop / max(target, 1.0e-6), 0.0, 1.0) * self._low_target_gate()

    def _reward_low_target_torso_pitch(self):
        torso_pitch = torch.abs(self.torso_projected_gravity[:, 0])
        min_pitch = float(self.cfg.rewards.pickup_torso_pitch_min)
        target_pitch = float(self.cfg.rewards.pickup_torso_pitch_target)
        pitch_score = torch.clamp(
            (torso_pitch - min_pitch) / max(target_pitch - min_pitch, 1.0e-6),
            0.0,
            1.0,
        )
        return pitch_score * self._low_target_gate()

    def _reward_low_target_forward_torso_pitch(self):
        torso_pitch = torch.clamp(-self.torso_projected_gravity[:, 0], min=0.0)
        min_pitch = float(self.cfg.rewards.pickup_torso_pitch_min)
        target_pitch = float(self.cfg.rewards.pickup_torso_pitch_target)
        pitch_score = torch.clamp(
            (torso_pitch - min_pitch) / max(target_pitch - min_pitch, 1.0e-6),
            0.0,
            1.0,
        )
        return pitch_score * self._low_target_gate()

    def _reward_low_target_torso_lateral_tilt(self):
        return torch.abs(self.torso_projected_gravity[:, 1]) * self._low_target_gate()

    def _reward_arm_leg_clearance(self):
        arm_parts = []
        for name in ("upper_arm_indices", "lower_arm_indices", "hand_indices", "end_effector_indices"):
            indices = getattr(self, name, None)
            if indices is not None and len(indices) > 0:
                arm_parts.append(indices)

        leg_parts = []
        for name in ("thigh_indices", "shank_indices", "feet_indices"):
            indices = getattr(self, name, None)
            if indices is not None and len(indices) > 0:
                leg_parts.append(indices)

        if len(arm_parts) == 0 or len(leg_parts) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        arm_indices = torch.cat(arm_parts)
        leg_indices = torch.cat(leg_parts)
        arm_pos = self.rigid_body_states[:, arm_indices, 0:3]
        leg_pos = self.rigid_body_states[:, leg_indices, 0:3]
        distances = torch.norm(arm_pos[:, :, None, :] - leg_pos[:, None, :, :], dim=-1)
        min_dist = torch.amin(distances, dim=(1, 2))
        margin = float(self.cfg.rewards.arm_leg_clearance_margin)
        return torch.clamp((margin - min_dist) / max(margin, 1.0e-6), min=0.0) * self._low_target_gate()


class GR2ReachPlantedPickupCfg(GR2ReachPickupHighCfg):
    """Same z goal as pickup/high, but planted feet and explicit squat rewards."""

    class reach(GR2ReachPickupHighCfg.reach):
        target_x_range = [0.16, 0.42]
        target_y_abs_range = [0.10, 0.34]
        target_z_range = [-0.36, 0.62]

    class rewards(GR2ReachPickupHighCfg.rewards):
        pickup_low_target_start_z = -0.08
        pickup_low_target_full_z = -0.32
        pickup_base_xy_deadband = 0.035
        pickup_feet_xy_deadband = 0.025
        pickup_feet_spread_allowance = 0.040
        pickup_feet_width_increase_target = 0.080
        pickup_feet_width_max = 0.36
        pickup_feet_midpoint_deadband = 0.035
        pickup_knee_flexion_target = 0.45
        pickup_knee_flexion_allowance = 0.10
        pickup_hip_flexion_target = 0.22
        pickup_base_height_drop_target = 0.16
        pickup_base_height_drop_allowance = 0.05
        pickup_lower_body_ground_clearance_min = 0.18
        pickup_lower_body_contact_force_threshold = 0.15
        pickup_ankle_pitch_target = 0.16
        pickup_yaw_twist_deadband = 0.08
        pickup_shoulder_drop_target = 0.14
        pickup_torso_pitch_min = 0.10
        pickup_torso_pitch_target = 0.38
        arm_leg_clearance_margin = 0.13

        class scales(GR2ReachPickupHighCfg.rewards.scales):
            reach_target_pos = 7.00
            reach_target_orient = 0.06
            reach_target_success = 1.15
            reach_target_stable = 3.20
            active_end_effector_still = -0.36
            head_look_at_target = 0.85

            base_still = -3.20
            base_xy_displacement = -4.00
            feet_xy_displacement = -3.00
            feet_speed_xy = -0.85
            low_target_knee_flexion = 4.50
            low_target_excess_knee_flexion = 0.00
            low_target_hip_flexion = 1.50
            low_target_base_height_drop = 2.00
            low_target_excess_base_height_drop = 0.00
            low_target_lower_body_ground_clearance = 0.00
            low_target_lower_body_contact = 0.00
            low_target_ankle_pitch = 0.80
            low_target_yaw_twist = -0.35
            low_target_squat_coherence = 0.80
            low_target_leg_asymmetry = -0.25
            low_target_abs_stance_width = 0.00
            low_target_feet_midpoint_drift = 0.00
            low_target_active_shoulder_drop = 0.80
            low_target_torso_pitch = 0.60
            low_target_torso_lateral_tilt = -0.30
            arm_leg_clearance = 0.00

            main_body_dof_pos = -0.45
            inactive_arm_still = -0.07

            base_height_offset_range = 0.45
            base_flat_orient = 1.90
            torso_flat_orient = 2.10

            action_diff = -0.48
            action_diff_diff = -0.11
            dof_acc = -0.032
            dof_tor = -0.032
            termination = -10.00

            limits_dof_pos_without_ankle = -4.50
            limits_dof_vel_without_ankle = -1.20
            limits_dof_tor = -0.30


class GR2ReachPlantedPickupCfgPPO(GR2ReachPickupHighCfgPPO, GR2ReachPlantedPickupCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2ReachPickupHighCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_pickup_from_expanded_stable"
        max_iterations = 3000
        save_interval = 100

        resume = True
        load_run = "Jun15_22-35-46_expanded_stable_headlook_from_robot_good"
        checkpoint = 10998

    class algorithm(GR2ReachPickupHighCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 2.e-5
        learning_rate_min = 8.e-6
        learning_rate_max = 1.5e-4
        schedule = "adaptive"
        desired_kl = 0.018

        storage_class = "RolloutStorage"

    class policy(GR2ReachPickupHighCfgPPO.policy):
        init_noise_std = [0.10] * GR2ReachPlantedPickupCfg.env.num_actions
