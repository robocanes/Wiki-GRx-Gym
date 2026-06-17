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

        self.episode_start_base_pos = self.base_pos.clone()
        self.episode_start_feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3].clone()

    def _reset_others(self, env_ids):
        super()._reset_others(env_ids)
        self.episode_start_base_pos[env_ids] = self.base_pos[env_ids]
        self.episode_start_feet_pos[env_ids] = self.rigid_body_states[
            env_ids[:, None], self.feet_indices, 0:3
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

    def _reward_feet_xy_displacement(self):
        feet_delta_xy = (
            self.rigid_body_states[:, self.feet_indices, 0:2]
            - self.episode_start_feet_pos[:, :, 0:2]
        )
        deadband = float(self.cfg.rewards.pickup_feet_xy_deadband)
        foot_disp = torch.sum(torch.clamp(torch.norm(feet_delta_xy, dim=2) - deadband, min=0.0), dim=1)
        return foot_disp

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
        pickup_knee_flexion_target = 0.45
        pickup_hip_flexion_target = 0.22
        pickup_base_height_drop_target = 0.16
        pickup_ankle_pitch_target = 0.16
        pickup_yaw_twist_deadband = 0.08

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
            low_target_hip_flexion = 1.50
            low_target_base_height_drop = 2.00
            low_target_ankle_pitch = 0.80
            low_target_yaw_twist = -0.35

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
