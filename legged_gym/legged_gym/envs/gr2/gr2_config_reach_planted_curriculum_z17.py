from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z14 import (
    GR2ReachPlantedCurriculumZ14Cfg,
    GR2ReachPlantedCurriculumZ14CfgPPO,
)


class GR2ReachPlantedCurriculumZ17Cfg(GR2ReachPlantedCurriculumZ14Cfg):
    """Train on the real 1-4 ft target-height band without over-favoring kneeling."""

    class reach(GR2ReachPlantedCurriculumZ14Cfg.reach):
        # Targets are sampled in the robot base frame. With the nominal 0.98 m
        # base height, 1-4 ft above ground maps to about -0.675..0.239 m.
        target_z_range = [-0.675, 0.239]
        target_edge_sample_prob = 0.68
        target_edge_z_band_fraction = 0.20
        target_edge_low_z_weight = 0.32
        target_edge_high_z_weight = 0.20
        target_edge_outer_y_weight = 0.20
        target_edge_forward_x_weight = 0.16

    class rewards(GR2ReachPlantedCurriculumZ14Cfg.rewards):
        pickup_low_target_start_z = -0.38
        pickup_low_target_full_z = -0.62

        pickup_torso_pitch_min = 0.04
        pickup_torso_pitch_target = 0.28
        pickup_lower_body_ground_clearance_min = 0.24
        pickup_lower_body_contact_force_threshold = 0.10
        pickup_base_height_drop_allowance = 0.025
        pickup_feet_width_increase_target = 0.036

        class scales(GR2ReachPlantedCurriculumZ14Cfg.rewards.scales):
            reach_target_pos = 38.00
            reach_target_success = 20.00
            reach_target_stable = 14.00

            low_target_forward_torso_pitch = 6.80
            low_target_active_shoulder_drop = 4.80

            low_target_knee_flexion = 0.95
            low_target_hip_flexion = 1.10
            low_target_base_height_drop = 0.55
            low_target_squat_coherence = 0.24
            low_target_ankle_pitch = 0.35

            low_target_lower_body_contact = -5.00
            low_target_lower_body_ground_clearance = -14.00
            low_target_excess_base_height_drop = -14.00
            low_target_stance_width = -1.45
            low_target_leg_asymmetry = -4.50
            low_target_torso_lateral_tilt = -4.80
            low_target_yaw_twist = -7.20

            feet_xy_displacement = -3.45
            feet_speed_xy = -1.00
            inactive_arm_still = -0.23
            arm_leg_clearance = -0.95
            collision = -0.55


class GR2ReachPlantedCurriculumZ17CfgPPO(
    GR2ReachPlantedCurriculumZ14CfgPPO,
    GR2ReachPlantedCurriculumZ17Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ14CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z17_real_height_band_from_z14_15100"
        max_iterations = 500
        save_interval = 50

        resume = True
        load_run = "Jun18_21-43-34_planted_curriculum_z14_anti_kneel_from_z10_15000_24576env"
        checkpoint = 15100

    class algorithm(GR2ReachPlantedCurriculumZ14CfgPPO.algorithm):
        learning_rate = 1.0e-6
        learning_rate_min = 5.0e-7
        learning_rate_max = 4.0e-6
        desired_kl = 0.003

    class policy(GR2ReachPlantedCurriculumZ14CfgPPO.policy):
        init_noise_std = [0.018] * GR2ReachPlantedCurriculumZ17Cfg.env.num_actions
