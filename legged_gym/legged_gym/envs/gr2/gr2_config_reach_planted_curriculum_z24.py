from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)


class GR2ReachPlantedCurriculumZ24Cfg(GR2ReachPlantedCurriculumZ17Cfg):
    """Stance-first fine-tune with reset exploration and direct foot-span penalties."""

    class reach(GR2ReachPlantedCurriculumZ17Cfg.reach):
        target_z_range = [-0.48, 0.239]
        target_y_abs_range = [0.03, 0.14]
        target_x_range = [0.18, 0.32]

        target_edge_sample_prob = 0.24
        target_edge_z_band_fraction = 0.10
        target_edge_y_band_fraction = 0.08
        target_edge_x_band_fraction = 0.10
        target_edge_low_z_weight = 0.08
        target_edge_high_z_weight = 0.20
        target_edge_outer_y_weight = 0.03
        target_edge_forward_x_weight = 0.12

    class rewards(GR2ReachPlantedCurriculumZ17Cfg.rewards):
        pickup_low_target_start_z = -0.24
        pickup_low_target_full_z = -0.44

        pickup_feet_xy_deadband = 0.018
        pickup_feet_spread_allowance = 0.012
        pickup_feet_width_increase_target = 0.010
        pickup_feet_width_max = 0.305
        pickup_feet_midpoint_deadband = 0.020

        pickup_knee_flexion_target = 0.24
        pickup_knee_flexion_allowance = 0.08
        pickup_base_height_drop_target = 0.07
        pickup_base_height_drop_allowance = 0.020
        pickup_lower_body_ground_clearance_min = 0.32
        pickup_lower_body_contact_force_threshold = 0.06
        pickup_torso_pitch_min = 0.10
        pickup_torso_pitch_target = 0.40
        arm_leg_clearance_margin = 0.24

        class scales(GR2ReachPlantedCurriculumZ17Cfg.rewards.scales):
            reach_target_pos = 18.00
            reach_target_success = 8.00
            reach_target_stable = 5.00

            low_target_forward_torso_pitch = 6.00
            low_target_active_shoulder_drop = 1.60

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


class GR2ReachPlantedCurriculumZ24CfgPPO(
    GR2ReachPlantedCurriculumZ17CfgPPO,
    GR2ReachPlantedCurriculumZ24Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ17CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z24_stance_first_from_z17_15599"
        max_iterations = 200
        save_interval = 25

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ17CfgPPO.algorithm):
        learning_rate = 5.0e-7
        learning_rate_min = 2.5e-7
        learning_rate_max = 2.0e-6
        desired_kl = 0.0018

    class policy(GR2ReachPlantedCurriculumZ17CfgPPO.policy):
        init_noise_std = [0.012] * GR2ReachPlantedCurriculumZ24Cfg.env.num_actions
        reset_loaded_std = True
