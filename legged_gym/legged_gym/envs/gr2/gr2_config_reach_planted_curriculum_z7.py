from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z6 import (
    GR2ReachPlantedCurriculumZ6Cfg,
    GR2ReachPlantedCurriculumZ6CfgPPO,
)


class GR2ReachPlantedCurriculumZ7Cfg(GR2ReachPlantedCurriculumZ6Cfg):
    """Wide-stance floor-reach refinement: keep the feet planted apart and fold forward instead of rotating."""

    class reach(GR2ReachPlantedCurriculumZ6Cfg.reach):
        # Keep targets close while learning the floor-reaching body shape.
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

    class rewards(GR2ReachPlantedCurriculumZ6Cfg.rewards):
        pickup_low_target_start_z = 0.08
        pickup_low_target_full_z = -0.42
        pickup_base_xy_deadband = 0.036
        pickup_feet_xy_deadband = 0.024
        pickup_feet_spread_allowance = 0.075
        pickup_feet_width_increase_target = 0.110
        pickup_knee_flexion_target = 0.78
        pickup_hip_flexion_target = 0.54
        pickup_base_height_drop_target = 0.32
        pickup_ankle_pitch_target = 0.31
        pickup_yaw_twist_deadband = 0.025
        pickup_shoulder_drop_target = 0.08
        pickup_torso_pitch_min = 0.00
        pickup_torso_pitch_target = 0.27

        class scales(GR2ReachPlantedCurriculumZ6Cfg.rewards.scales):
            reach_target_pos = 8.60
            reach_target_success = 1.55
            reach_target_stable = 3.90
            head_look_at_target = 1.05

            base_still = -3.60
            base_xy_displacement = -3.20
            feet_xy_displacement = -2.10
            feet_speed_xy = -0.85
            low_target_knee_flexion = 7.20
            low_target_hip_flexion = 7.20
            low_target_base_height_drop = 4.80
            low_target_ankle_pitch = 2.20
            low_target_yaw_twist = -4.20
            low_target_squat_coherence = 2.40
            low_target_leg_asymmetry = -1.45
            low_target_active_shoulder_drop = 8.50
            low_target_torso_pitch = 8.00
            low_target_torso_lateral_tilt = -2.20
            low_target_stance_width = 4.80

            # Low floor reach needs controlled forward fold; don't over-reward vertical torso.
            base_height_offset_range = 0.12
            base_flat_orient = 0.20
            torso_flat_orient = 0.08

            main_body_dof_pos = -0.15
            action_diff = -0.42
            action_diff_diff = -0.10


class GR2ReachPlantedCurriculumZ7CfgPPO(
    GR2ReachPlantedCurriculumZ6CfgPPO,
    GR2ReachPlantedCurriculumZ7Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ6CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z7_wide_squat_from_z6_14045"
        max_iterations = 260
        save_interval = 50

        resume = True
        load_run = "Jun17_17-50-10_planted_curriculum_z6_soft_floor_fold_from_z5_13746"
        checkpoint = 14045

    class algorithm(GR2ReachPlantedCurriculumZ6CfgPPO.algorithm):
        learning_rate = 5.0e-6
        learning_rate_min = 1.8e-6
        learning_rate_max = 3.2e-5
        desired_kl = 0.008

    class policy(GR2ReachPlantedCurriculumZ6CfgPPO.policy):
        init_noise_std = [0.040] * GR2ReachPlantedCurriculumZ7Cfg.env.num_actions
