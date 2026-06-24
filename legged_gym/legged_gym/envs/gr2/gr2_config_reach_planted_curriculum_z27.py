from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z23 import (
    GR2ReachPlantedCurriculumZ23Cfg,
    GR2ReachPlantedCurriculumZ23CfgPPO,
)


class GR2ReachPlantedCurriculumZ27Cfg(GR2ReachPlantedCurriculumZ23Cfg):
    """Balance-over-feet branch: lean forward while keeping the upper-body mass proxy in support."""

    class reach(GR2ReachPlantedCurriculumZ23Cfg.reach):
        target_z_range = [-0.52, 0.239]
        target_y_abs_range = [0.03, 0.15]
        target_x_range = [0.18, 0.34]

        target_edge_sample_prob = 0.26
        target_edge_z_band_fraction = 0.10
        target_edge_y_band_fraction = 0.08
        target_edge_x_band_fraction = 0.10
        target_edge_low_z_weight = 0.10
        target_edge_high_z_weight = 0.20
        target_edge_outer_y_weight = 0.03
        target_edge_forward_x_weight = 0.12

    class rewards(GR2ReachPlantedCurriculumZ23Cfg.rewards):
        pickup_low_target_start_z = -0.26
        pickup_low_target_full_z = -0.48

        pickup_feet_xy_deadband = 0.018
        pickup_feet_spread_allowance = 0.012
        pickup_feet_width_increase_target = 0.010
        pickup_feet_width_max = 0.315
        pickup_feet_midpoint_deadband = 0.020

        pickup_support_foot_half_length = 0.15
        pickup_support_foot_half_width = 0.060
        pickup_support_margin = 0.010
        pickup_support_error_scale = 0.06
        pickup_com_forward_min = 0.035
        pickup_com_forward_target = 0.14

        pickup_knee_flexion_target = 0.24
        pickup_knee_flexion_allowance = 0.08
        pickup_base_height_drop_target = 0.08
        pickup_base_height_drop_allowance = 0.020
        pickup_lower_body_ground_clearance_min = 0.32
        pickup_lower_body_contact_force_threshold = 0.06
        pickup_torso_pitch_min = 0.12
        pickup_torso_pitch_target = 0.48
        arm_leg_clearance_margin = 0.24

        class scales(GR2ReachPlantedCurriculumZ23Cfg.rewards.scales):
            reach_target_pos = 18.00
            reach_target_success = 8.00
            reach_target_stable = 5.00

            low_target_forward_torso_pitch = 11.00
            low_target_com_forward = 7.00
            low_target_com_support = -36.00
            low_target_active_shoulder_drop = 1.40

            low_target_knee_flexion = -0.10
            low_target_excess_knee_flexion = -10.00
            low_target_hip_flexion = 0.00
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.00

            low_target_lower_body_contact = -14.00
            low_target_lower_body_ground_clearance = -30.00
            low_target_excess_base_height_drop = -28.00
            low_target_stance_width = -8.00
            low_target_abs_stance_width = -30.00
            low_target_feet_midpoint_drift = -24.00
            low_target_leg_asymmetry = -8.50
            low_target_torso_lateral_tilt = -9.00
            low_target_yaw_twist = -11.00

            feet_xy_displacement = -7.50
            feet_speed_xy = -1.60
            inactive_arm_still = -0.55
            arm_leg_clearance = -7.00
            collision = -1.30


class GR2ReachPlantedCurriculumZ27CfgPPO(
    GR2ReachPlantedCurriculumZ23CfgPPO,
    GR2ReachPlantedCurriculumZ27Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ23CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z27_com_forward_from_z17_15599"
        max_iterations = 180
        save_interval = 25

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ23CfgPPO.algorithm):
        learning_rate = 5.0e-7
        learning_rate_min = 2.5e-7
        learning_rate_max = 2.0e-6
        desired_kl = 0.0018

    class policy(GR2ReachPlantedCurriculumZ23CfgPPO.policy):
        init_noise_std = [0.012] * GR2ReachPlantedCurriculumZ27Cfg.env.num_actions
        reset_loaded_std = True
