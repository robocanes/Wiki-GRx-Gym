from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)


class GR2ReachPlantedCurriculumZ25Cfg(GR2ReachPlantedCurriculumZ17Cfg):
    """Posture calibration: near/mid-low targets with dominant planted-feet costs."""

    class reach(GR2ReachPlantedCurriculumZ17Cfg.reach):
        target_z_range = [-0.34, 0.18]
        target_y_abs_range = [0.02, 0.10]
        target_x_range = [0.18, 0.28]

        target_edge_sample_prob = 0.12
        target_edge_z_band_fraction = 0.08
        target_edge_y_band_fraction = 0.06
        target_edge_x_band_fraction = 0.08
        target_edge_low_z_weight = 0.04
        target_edge_high_z_weight = 0.12
        target_edge_outer_y_weight = 0.02
        target_edge_forward_x_weight = 0.08

    class rewards(GR2ReachPlantedCurriculumZ17Cfg.rewards):
        pickup_low_target_start_z = -0.18
        pickup_low_target_full_z = -0.32

        pickup_feet_xy_deadband = 0.010
        pickup_feet_spread_allowance = 0.004
        pickup_feet_width_increase_target = 0.006
        pickup_feet_width_max = 0.285
        pickup_feet_midpoint_deadband = 0.010

        pickup_knee_flexion_target = 0.18
        pickup_knee_flexion_allowance = 0.06
        pickup_base_height_drop_target = 0.05
        pickup_base_height_drop_allowance = 0.015
        pickup_lower_body_ground_clearance_min = 0.34
        pickup_lower_body_contact_force_threshold = 0.05
        pickup_torso_pitch_min = 0.06
        pickup_torso_pitch_target = 0.30
        arm_leg_clearance_margin = 0.24

        class scales(GR2ReachPlantedCurriculumZ17Cfg.rewards.scales):
            reach_target_pos = 4.00
            reach_target_success = 1.00
            reach_target_stable = 1.00

            low_target_forward_torso_pitch = 2.00
            low_target_active_shoulder_drop = 0.50

            low_target_knee_flexion = -0.20
            low_target_excess_knee_flexion = -12.00
            low_target_hip_flexion = 0.00
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.00

            low_target_lower_body_contact = -16.00
            low_target_lower_body_ground_clearance = -36.00
            low_target_excess_base_height_drop = -36.00
            low_target_stance_width = -24.00
            low_target_abs_stance_width = -160.00
            low_target_feet_midpoint_drift = -140.00
            low_target_leg_asymmetry = -12.00
            low_target_torso_lateral_tilt = -12.00
            low_target_yaw_twist = -14.00

            feet_xy_displacement = -22.00
            feet_speed_xy = -2.20
            inactive_arm_still = -0.60
            arm_leg_clearance = -8.00
            collision = -1.50


class GR2ReachPlantedCurriculumZ25CfgPPO(
    GR2ReachPlantedCurriculumZ17CfgPPO,
    GR2ReachPlantedCurriculumZ25Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ17CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z25_posture_calibration_from_z17_15599"
        max_iterations = 120
        save_interval = 10

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ17CfgPPO.algorithm):
        learning_rate = 4.0e-7
        learning_rate_min = 2.0e-7
        learning_rate_max = 1.5e-6
        desired_kl = 0.0015

    class policy(GR2ReachPlantedCurriculumZ17CfgPPO.policy):
        init_noise_std = [0.008] * GR2ReachPlantedCurriculumZ25Cfg.env.num_actions
        reset_loaded_std = True
