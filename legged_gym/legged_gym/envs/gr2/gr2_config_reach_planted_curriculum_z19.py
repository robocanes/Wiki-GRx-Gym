from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)


class GR2ReachPlantedCurriculumZ19Cfg(GR2ReachPlantedCurriculumZ17Cfg):
    """Reduce low-and-outer targets that encourage side-squat arm-through-leg solutions."""

    class reach(GR2ReachPlantedCurriculumZ17Cfg.reach):
        target_z_range = [-0.675, 0.239]
        target_y_abs_range = [0.06, 0.22]
        target_x_range = [0.12, 0.36]

        target_edge_sample_prob = 0.48
        target_edge_z_band_fraction = 0.16
        target_edge_y_band_fraction = 0.16
        target_edge_x_band_fraction = 0.16
        target_edge_low_z_weight = 0.22
        target_edge_high_z_weight = 0.20
        target_edge_outer_y_weight = 0.10
        target_edge_forward_x_weight = 0.18

    class rewards(GR2ReachPlantedCurriculumZ17Cfg.rewards):
        pickup_low_target_start_z = -0.36
        pickup_low_target_full_z = -0.60

        pickup_knee_flexion_target = 0.30
        pickup_knee_flexion_allowance = 0.10
        pickup_base_height_drop_target = 0.10
        pickup_base_height_drop_allowance = 0.030
        pickup_lower_body_ground_clearance_min = 0.30
        pickup_lower_body_contact_force_threshold = 0.08
        pickup_torso_pitch_min = 0.08
        pickup_torso_pitch_target = 0.40
        pickup_feet_width_increase_target = 0.025
        arm_leg_clearance_margin = 0.20

        class scales(GR2ReachPlantedCurriculumZ17Cfg.rewards.scales):
            reach_target_pos = 38.00
            reach_target_success = 20.00
            reach_target_stable = 14.00

            low_target_forward_torso_pitch = 7.40
            low_target_active_shoulder_drop = 3.60

            low_target_knee_flexion = 0.00
            low_target_excess_knee_flexion = -8.00
            low_target_hip_flexion = 0.35
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.15

            low_target_lower_body_contact = -10.00
            low_target_lower_body_ground_clearance = -22.00
            low_target_excess_base_height_drop = -18.00
            low_target_stance_width = -2.70
            low_target_leg_asymmetry = -6.00
            low_target_torso_lateral_tilt = -7.50
            low_target_yaw_twist = -9.00

            feet_xy_displacement = -4.10
            feet_speed_xy = -1.10
            inactive_arm_still = -0.34
            arm_leg_clearance = -3.80
            collision = -0.90


class GR2ReachPlantedCurriculumZ19CfgPPO(
    GR2ReachPlantedCurriculumZ17CfgPPO,
    GR2ReachPlantedCurriculumZ19Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ17CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z19_forward_band_from_z17_15599"
        max_iterations = 400
        save_interval = 50

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ17CfgPPO.algorithm):
        learning_rate = 8.0e-7
        learning_rate_min = 4.0e-7
        learning_rate_max = 3.0e-6
        desired_kl = 0.0025

    class policy(GR2ReachPlantedCurriculumZ17CfgPPO.policy):
        init_noise_std = [0.014] * GR2ReachPlantedCurriculumZ19Cfg.env.num_actions
