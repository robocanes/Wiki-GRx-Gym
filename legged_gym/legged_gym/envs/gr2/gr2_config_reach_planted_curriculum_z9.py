from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z7 import (
    GR2ReachPlantedCurriculumZ7Cfg,
    GR2ReachPlantedCurriculumZ7CfgPPO,
)


class GR2ReachPlantedCurriculumZ9Cfg(GR2ReachPlantedCurriculumZ7Cfg):
    """Reopen reaching after Z8 cleaned up foot spread and inactive-arm exploits."""

    class reach(GR2ReachPlantedCurriculumZ7Cfg.reach):
        target_x_range = [0.10, 0.30]
        target_y_abs_range = [0.08, 0.25]
        target_z_range = [-0.52, 0.00]

        target_edge_sample_prob = 0.92
        target_edge_z_band_fraction = 0.36
        target_edge_y_band_fraction = 0.22
        target_edge_x_band_fraction = 0.16
        target_edge_low_z_weight = 0.88
        target_edge_high_z_weight = 0.02
        target_edge_outer_y_weight = 0.08
        target_edge_forward_x_weight = 0.06

    class rewards(GR2ReachPlantedCurriculumZ7Cfg.rewards):
        pickup_low_target_start_z = 0.08
        pickup_low_target_full_z = -0.42

        pickup_base_xy_deadband = 0.034
        pickup_feet_xy_deadband = 0.022
        pickup_feet_spread_allowance = 0.045
        pickup_feet_width_increase_target = 0.055

        pickup_knee_flexion_target = 0.66
        pickup_hip_flexion_target = 0.46
        pickup_base_height_drop_target = 0.27
        pickup_ankle_pitch_target = 0.26
        pickup_yaw_twist_deadband = 0.020
        pickup_shoulder_drop_target = 0.08
        pickup_torso_pitch_min = 0.02
        pickup_torso_pitch_target = 0.22

        class scales(GR2ReachPlantedCurriculumZ7Cfg.rewards.scales):
            reach_target_pos = 30.00
            reach_target_success = 14.00
            reach_target_stable = 12.00
            head_look_at_target = 1.50

            base_still = -3.60
            base_xy_displacement = -3.80
            feet_xy_displacement = -2.80
            feet_speed_xy = -0.90
            inactive_arm_still = -0.14

            low_target_knee_flexion = 2.80
            low_target_hip_flexion = 2.80
            low_target_base_height_drop = 2.00
            low_target_ankle_pitch = 0.90
            low_target_yaw_twist = -6.00
            low_target_squat_coherence = 0.80
            low_target_leg_asymmetry = -3.20
            low_target_active_shoulder_drop = 4.20
            low_target_torso_pitch = 0.00
            low_target_forward_torso_pitch = 8.50
            low_target_torso_lateral_tilt = -3.80
            low_target_stance_width = 0.00

            base_height_offset_range = 0.10
            base_flat_orient = 0.35
            torso_flat_orient = 0.18

            main_body_dof_pos = -0.20
            action_diff = -0.50
            action_diff_diff = -0.12


class GR2ReachPlantedCurriculumZ9CfgPPO(
    GR2ReachPlantedCurriculumZ7CfgPPO,
    GR2ReachPlantedCurriculumZ9Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ7CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z9_reach_rebalance_from_z8_14350"
        max_iterations = 1500
        save_interval = 50

        resume = True
        load_run = "Jun18_15-14-09_planted_curriculum_z8_forward_pitch_from_z7_14304_24576env"
        checkpoint = 14350

    class algorithm(GR2ReachPlantedCurriculumZ7CfgPPO.algorithm):
        learning_rate = 2.5e-6
        learning_rate_min = 1.2e-6
        learning_rate_max = 1.0e-5
        desired_kl = 0.006

    class policy(GR2ReachPlantedCurriculumZ7CfgPPO.policy):
        init_noise_std = [0.030] * GR2ReachPlantedCurriculumZ9Cfg.env.num_actions
