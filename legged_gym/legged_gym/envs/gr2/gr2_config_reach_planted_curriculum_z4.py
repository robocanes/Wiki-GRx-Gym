from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z3 import (
    GR2ReachPlantedCurriculumZ3Cfg,
    GR2ReachPlantedCurriculumZ3CfgPPO,
)


class GR2ReachPlantedCurriculumZ4Cfg(GR2ReachPlantedCurriculumZ3Cfg):
    """Low-reach squat stage, initialized from the good Z3 upper/lower policy."""

    class reach(GR2ReachPlantedCurriculumZ3Cfg.reach):
        # Do not extend forward x yet; teach lower targets through a planted squat.
        target_x_range = [0.16, 0.42]
        target_y_abs_range = [0.10, 0.36]
        target_z_range = [-0.42, 0.78]

        target_edge_sample_prob = 0.84
        target_edge_z_band_fraction = 0.28
        target_edge_y_band_fraction = 0.26
        target_edge_x_band_fraction = 0.22
        target_edge_low_z_weight = 0.56
        target_edge_high_z_weight = 0.28
        target_edge_outer_y_weight = 0.11
        target_edge_forward_x_weight = 0.05

    class rewards(GR2ReachPlantedCurriculumZ3Cfg.rewards):
        pickup_low_target_start_z = 0.10
        pickup_low_target_full_z = -0.38
        pickup_base_xy_deadband = 0.050
        pickup_feet_xy_deadband = 0.040
        pickup_knee_flexion_target = 0.62
        pickup_hip_flexion_target = 0.31
        pickup_base_height_drop_target = 0.22
        pickup_ankle_pitch_target = 0.22
        pickup_yaw_twist_deadband = 0.055

        class scales(GR2ReachPlantedCurriculumZ3Cfg.rewards.scales):
            reach_target_pos = 7.65
            reach_target_success = 1.35
            reach_target_stable = 3.55
            head_look_at_target = 1.00

            base_still = -3.25
            base_xy_displacement = -1.75
            feet_xy_displacement = -1.30
            feet_speed_xy = -0.35
            low_target_knee_flexion = 8.20
            low_target_hip_flexion = 3.30
            low_target_base_height_drop = 4.80
            low_target_ankle_pitch = 1.55
            low_target_yaw_twist = -1.10

            base_height_offset_range = 0.32
            base_flat_orient = 1.95
            torso_flat_orient = 2.15

            action_diff = -0.45
            action_diff_diff = -0.105


class GR2ReachPlantedCurriculumZ4CfgPPO(
    GR2ReachPlantedCurriculumZ3CfgPPO,
    GR2ReachPlantedCurriculumZ4Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ3CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z4_squat_from_z3_12698"
        max_iterations = 700
        save_interval = 100

        resume = True
        load_run = "Jun16_22-40-14_planted_curriculum_z3_ground_high_from_z2_12099"
        checkpoint = 12698

    class algorithm(GR2ReachPlantedCurriculumZ3CfgPPO.algorithm):
        learning_rate = 9.0e-6
        learning_rate_min = 3.e-6
        learning_rate_max = 6.0e-5
        desired_kl = 0.012

    class policy(GR2ReachPlantedCurriculumZ3CfgPPO.policy):
        init_noise_std = [0.060] * GR2ReachPlantedCurriculumZ4Cfg.env.num_actions
