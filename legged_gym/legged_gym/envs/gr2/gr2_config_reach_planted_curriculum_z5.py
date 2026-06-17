from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z4 import (
    GR2ReachPlantedCurriculumZ4Cfg,
    GR2ReachPlantedCurriculumZ4CfgPPO,
)


class GR2ReachPlantedCurriculumZ5Cfg(GR2ReachPlantedCurriculumZ4Cfg):
    """Low-target squat refinement, initialized from the finished Z4 policy."""

    class reach(GR2ReachPlantedCurriculumZ4Cfg.reach):
        # Temporarily narrow x/y so the policy learns to squat down before re-expanding.
        target_x_range = [0.14, 0.34]
        target_y_abs_range = [0.08, 0.26]
        target_z_range = [-0.46, 0.08]

        target_edge_sample_prob = 0.90
        target_edge_z_band_fraction = 0.34
        target_edge_y_band_fraction = 0.24
        target_edge_x_band_fraction = 0.18
        target_edge_low_z_weight = 0.76
        target_edge_high_z_weight = 0.08
        target_edge_outer_y_weight = 0.10
        target_edge_forward_x_weight = 0.06

    class rewards(GR2ReachPlantedCurriculumZ4Cfg.rewards):
        pickup_low_target_start_z = 0.12
        pickup_low_target_full_z = -0.40
        pickup_base_xy_deadband = 0.040
        pickup_feet_xy_deadband = 0.028
        pickup_knee_flexion_target = 0.72
        pickup_hip_flexion_target = 0.36
        pickup_base_height_drop_target = 0.28
        pickup_ankle_pitch_target = 0.26
        pickup_yaw_twist_deadband = 0.040

        class scales(GR2ReachPlantedCurriculumZ4Cfg.rewards.scales):
            reach_target_pos = 8.10
            reach_target_success = 1.45
            reach_target_stable = 3.70
            head_look_at_target = 1.05

            base_still = -3.50
            base_xy_displacement = -2.20
            feet_xy_displacement = -2.30
            feet_speed_xy = -0.55
            low_target_knee_flexion = 8.80
            low_target_hip_flexion = 3.85
            low_target_base_height_drop = 6.30
            low_target_ankle_pitch = 2.15
            low_target_yaw_twist = -1.75
            low_target_squat_coherence = 3.20
            low_target_leg_asymmetry = -1.10

            base_height_offset_range = 0.22
            base_flat_orient = 2.05
            torso_flat_orient = 2.25

            action_diff = -0.43
            action_diff_diff = -0.10


class GR2ReachPlantedCurriculumZ5CfgPPO(
    GR2ReachPlantedCurriculumZ4CfgPPO,
    GR2ReachPlantedCurriculumZ5Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ4CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z5_low_squat_from_z4_13397"
        max_iterations = 350
        save_interval = 50

        resume = True
        load_run = "Jun17_14-02-30_planted_curriculum_z4_squat_from_z3_12698"
        checkpoint = 13397

    class algorithm(GR2ReachPlantedCurriculumZ4CfgPPO.algorithm):
        learning_rate = 7.0e-6
        learning_rate_min = 2.5e-6
        learning_rate_max = 4.5e-5
        desired_kl = 0.010

    class policy(GR2ReachPlantedCurriculumZ4CfgPPO.policy):
        init_noise_std = [0.050] * GR2ReachPlantedCurriculumZ5Cfg.env.num_actions
