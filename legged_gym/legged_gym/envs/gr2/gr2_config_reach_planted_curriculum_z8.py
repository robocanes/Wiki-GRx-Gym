from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z7 import (
    GR2ReachPlantedCurriculumZ7Cfg,
    GR2ReachPlantedCurriculumZ7CfgPPO,
)


class GR2ReachPlantedCurriculumZ8Cfg(GR2ReachPlantedCurriculumZ7Cfg):
    """Recover from Z7's split/back-lean attractor by prioritizing actual target reach."""

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

        pickup_base_xy_deadband = 0.030
        pickup_feet_xy_deadband = 0.018
        pickup_feet_spread_allowance = 0.030
        pickup_feet_width_increase_target = 0.055

        pickup_knee_flexion_target = 0.62
        pickup_hip_flexion_target = 0.42
        pickup_base_height_drop_target = 0.24
        pickup_ankle_pitch_target = 0.24
        pickup_yaw_twist_deadband = 0.020
        pickup_shoulder_drop_target = 0.08
        pickup_torso_pitch_min = 0.02
        pickup_torso_pitch_target = 0.22

        class scales(GR2ReachPlantedCurriculumZ7Cfg.rewards.scales):
            reach_target_pos = 16.00
            reach_target_success = 5.00
            reach_target_stable = 7.50
            head_look_at_target = 1.20

            base_still = -4.40
            base_xy_displacement = -5.80
            feet_xy_displacement = -5.60
            feet_speed_xy = -1.40
            inactive_arm_still = -0.28

            low_target_knee_flexion = 2.40
            low_target_hip_flexion = 2.40
            low_target_base_height_drop = 1.60
            low_target_ankle_pitch = 0.80
            low_target_yaw_twist = -6.00
            low_target_squat_coherence = 0.80
            low_target_leg_asymmetry = -3.20
            low_target_active_shoulder_drop = 2.80
            low_target_torso_pitch = 0.00
            low_target_forward_torso_pitch = 4.20
            low_target_torso_lateral_tilt = -5.00
            low_target_stance_width = 0.00

            base_height_offset_range = 0.08
            base_flat_orient = 0.35
            torso_flat_orient = 0.18

            main_body_dof_pos = -0.20
            action_diff = -0.50
            action_diff_diff = -0.12


class GR2ReachPlantedCurriculumZ8CfgPPO(
    GR2ReachPlantedCurriculumZ7CfgPPO,
    GR2ReachPlantedCurriculumZ8Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ7CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z8_forward_pitch_from_z7_14304"
        max_iterations = 3000
        save_interval = 50

        resume = True
        load_run = "Jun17_21-28-50_planted_curriculum_z7_wide_squat_from_z6_14045"
        checkpoint = 14304

    class algorithm(GR2ReachPlantedCurriculumZ7CfgPPO.algorithm):
        learning_rate = 3.5e-6
        learning_rate_min = 1.2e-6
        learning_rate_max = 1.5e-5
        desired_kl = 0.006

    class policy(GR2ReachPlantedCurriculumZ7CfgPPO.policy):
        init_noise_std = [0.030] * GR2ReachPlantedCurriculumZ8Cfg.env.num_actions
