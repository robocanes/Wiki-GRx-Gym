from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z27 import (
    GR2ReachPlantedCurriculumZ27Cfg,
    GR2ReachPlantedCurriculumZ27CfgPPO,
)


class GR2ReachPlantedCurriculumZ29Cfg(GR2ReachPlantedCurriculumZ27Cfg):
    """Posture-priority branch from the more compact Z5 actor."""

    class reach(GR2ReachPlantedCurriculumZ27Cfg.reach):
        target_z_range = [-0.42, 0.18]
        target_y_abs_range = [0.02, 0.11]
        target_x_range = [0.20, 0.34]

        target_edge_sample_prob = 0.10
        target_edge_z_band_fraction = 0.08
        target_edge_y_band_fraction = 0.05
        target_edge_x_band_fraction = 0.08
        target_edge_low_z_weight = 0.03
        target_edge_high_z_weight = 0.12
        target_edge_outer_y_weight = 0.01
        target_edge_forward_x_weight = 0.10

    class rewards(GR2ReachPlantedCurriculumZ27Cfg.rewards):
        pickup_low_target_start_z = -0.18
        pickup_low_target_full_z = -0.38

        pickup_feet_xy_deadband = 0.012
        pickup_feet_spread_allowance = 0.006
        pickup_feet_width_increase_target = 0.010
        pickup_feet_width_max = 0.300
        pickup_feet_midpoint_deadband = 0.012

        pickup_support_foot_half_length = 0.15
        pickup_support_foot_half_width = 0.060
        pickup_support_margin = 0.010
        pickup_support_error_scale = 0.045
        pickup_com_forward_min = 0.020
        pickup_com_forward_target = 0.11

        pickup_knee_flexion_target = 0.18
        pickup_knee_flexion_allowance = 0.06
        pickup_base_height_drop_target = 0.05
        pickup_base_height_drop_allowance = 0.015
        pickup_lower_body_ground_clearance_min = 0.34
        pickup_lower_body_contact_force_threshold = 0.06
        pickup_torso_pitch_min = 0.08
        pickup_torso_pitch_target = 0.52
        arm_leg_clearance_margin = 0.24

        class scales(GR2ReachPlantedCurriculumZ27Cfg.rewards.scales):
            reach_target_pos = 5.00
            reach_target_success = 1.50
            reach_target_stable = 1.50

            low_target_forward_torso_pitch = 34.00
            low_target_com_forward = 24.00
            low_target_com_support = -64.00
            low_target_active_shoulder_drop = 0.80

            low_target_knee_flexion = -0.30
            low_target_excess_knee_flexion = -18.00
            low_target_hip_flexion = 0.00
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.00

            low_target_lower_body_contact = -18.00
            low_target_lower_body_ground_clearance = -42.00
            low_target_excess_base_height_drop = -42.00
            low_target_stance_width = -24.00
            low_target_abs_stance_width = -120.00
            low_target_feet_midpoint_drift = -120.00
            low_target_leg_asymmetry = -14.00
            low_target_torso_lateral_tilt = -14.00
            low_target_yaw_twist = -16.00

            feet_xy_displacement = -24.00
            feet_speed_xy = -2.40
            inactive_arm_still = -0.70
            arm_leg_clearance = -10.00
            collision = -1.80


class GR2ReachPlantedCurriculumZ29CfgPPO(
    GR2ReachPlantedCurriculumZ27CfgPPO,
    GR2ReachPlantedCurriculumZ29Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ27CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z29_posture_first_from_z5_13746"
        max_iterations = 120
        save_interval = 10

        resume = True
        load_run = "Jun17_16-09-49_planted_curriculum_z5_low_squat_from_z4_13397"
        checkpoint = 13746

    class algorithm(GR2ReachPlantedCurriculumZ27CfgPPO.algorithm):
        learning_rate = 3.0e-7
        learning_rate_min = 1.5e-7
        learning_rate_max = 1.2e-6
        desired_kl = 0.0012

    class policy(GR2ReachPlantedCurriculumZ27CfgPPO.policy):
        init_noise_std = [0.008] * GR2ReachPlantedCurriculumZ29Cfg.env.num_actions
        reset_loaded_std = True
