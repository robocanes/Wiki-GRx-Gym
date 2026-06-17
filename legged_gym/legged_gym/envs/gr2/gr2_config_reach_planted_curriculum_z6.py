from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z5 import (
    GR2ReachPlantedCurriculumZ5Cfg,
    GR2ReachPlantedCurriculumZ5CfgPPO,
)


class GR2ReachPlantedCurriculumZ6Cfg(GR2ReachPlantedCurriculumZ5Cfg):
    """Floor-reach refinement: fold the hips/torso enough to lower the active shoulder."""

    class reach(GR2ReachPlantedCurriculumZ5Cfg.reach):
        # Keep targets close while learning the floor-reaching body shape.
        target_x_range = [0.12, 0.30]
        target_y_abs_range = [0.06, 0.22]
        target_z_range = [-0.48, 0.02]

        target_edge_sample_prob = 0.92
        target_edge_z_band_fraction = 0.36
        target_edge_y_band_fraction = 0.22
        target_edge_x_band_fraction = 0.16
        target_edge_low_z_weight = 0.82
        target_edge_high_z_weight = 0.04
        target_edge_outer_y_weight = 0.08
        target_edge_forward_x_weight = 0.06

    class rewards(GR2ReachPlantedCurriculumZ5Cfg.rewards):
        pickup_low_target_start_z = 0.08
        pickup_low_target_full_z = -0.42
        pickup_base_xy_deadband = 0.036
        pickup_feet_xy_deadband = 0.024
        pickup_knee_flexion_target = 0.74
        pickup_hip_flexion_target = 0.48
        pickup_base_height_drop_target = 0.30
        pickup_ankle_pitch_target = 0.28
        pickup_yaw_twist_deadband = 0.035
        pickup_shoulder_drop_target = 0.08
        pickup_torso_pitch_min = 0.00
        pickup_torso_pitch_target = 0.22

        class scales(GR2ReachPlantedCurriculumZ5Cfg.rewards.scales):
            reach_target_pos = 8.60
            reach_target_success = 1.55
            reach_target_stable = 3.90
            head_look_at_target = 1.05

            base_still = -3.60
            base_xy_displacement = -2.60
            feet_xy_displacement = -2.75
            feet_speed_xy = -0.70
            low_target_knee_flexion = 6.80
            low_target_hip_flexion = 5.80
            low_target_base_height_drop = 4.50
            low_target_ankle_pitch = 1.70
            low_target_yaw_twist = -2.60
            low_target_squat_coherence = 1.70
            low_target_leg_asymmetry = -1.25
            low_target_active_shoulder_drop = 8.00
            low_target_torso_pitch = 7.00
            low_target_torso_lateral_tilt = -1.60

            # Low floor reach needs controlled forward fold; don't over-reward vertical torso.
            base_height_offset_range = 0.12
            base_flat_orient = 0.35
            torso_flat_orient = 0.15

            main_body_dof_pos = -0.15
            action_diff = -0.42
            action_diff_diff = -0.10


class GR2ReachPlantedCurriculumZ6CfgPPO(
    GR2ReachPlantedCurriculumZ5CfgPPO,
    GR2ReachPlantedCurriculumZ6Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ5CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z6_soft_floor_fold_from_z5_13746"
        max_iterations = 300
        save_interval = 50

        resume = True
        load_run = "Jun17_16-09-49_planted_curriculum_z5_low_squat_from_z4_13397"
        checkpoint = 13746

    class algorithm(GR2ReachPlantedCurriculumZ5CfgPPO.algorithm):
        learning_rate = 6.0e-6
        learning_rate_min = 2.0e-6
        learning_rate_max = 3.8e-5
        desired_kl = 0.009

    class policy(GR2ReachPlantedCurriculumZ5CfgPPO.policy):
        init_noise_std = [0.045] * GR2ReachPlantedCurriculumZ6Cfg.env.num_actions
