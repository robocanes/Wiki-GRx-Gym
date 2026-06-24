from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)


class GR2ReachPlantedCurriculumZ21Cfg(GR2ReachPlantedCurriculumZ17Cfg):
    """Learn a compact forward reach before reintroducing the deepest target edge."""

    class reach(GR2ReachPlantedCurriculumZ17Cfg.reach):
        target_z_range = [-0.50, 0.239]
        target_y_abs_range = [0.04, 0.16]
        target_x_range = [0.18, 0.32]

        target_edge_sample_prob = 0.34
        target_edge_z_band_fraction = 0.12
        target_edge_y_band_fraction = 0.10
        target_edge_x_band_fraction = 0.12
        target_edge_low_z_weight = 0.12
        target_edge_high_z_weight = 0.22
        target_edge_outer_y_weight = 0.05
        target_edge_forward_x_weight = 0.18

    class rewards(GR2ReachPlantedCurriculumZ17Cfg.rewards):
        pickup_low_target_start_z = -0.26
        pickup_low_target_full_z = -0.46

        pickup_knee_flexion_target = 0.28
        pickup_knee_flexion_allowance = 0.10
        pickup_base_height_drop_target = 0.09
        pickup_base_height_drop_allowance = 0.030
        pickup_lower_body_ground_clearance_min = 0.30
        pickup_lower_body_contact_force_threshold = 0.08
        pickup_torso_pitch_min = 0.10
        pickup_torso_pitch_target = 0.42
        pickup_feet_width_increase_target = 0.016
        arm_leg_clearance_margin = 0.22

        class scales(GR2ReachPlantedCurriculumZ17Cfg.rewards.scales):
            reach_target_pos = 30.00
            reach_target_success = 16.00
            reach_target_stable = 10.00

            low_target_forward_torso_pitch = 8.00
            low_target_active_shoulder_drop = 2.80

            low_target_knee_flexion = 0.00
            low_target_excess_knee_flexion = -10.00
            low_target_hip_flexion = 0.20
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.08

            low_target_lower_body_contact = -12.00
            low_target_lower_body_ground_clearance = -26.00
            low_target_excess_base_height_drop = -22.00
            low_target_stance_width = -3.80
            low_target_leg_asymmetry = -6.80
            low_target_torso_lateral_tilt = -8.00
            low_target_yaw_twist = -9.80

            feet_xy_displacement = -4.80
            feet_speed_xy = -1.20
            inactive_arm_still = -0.42
            arm_leg_clearance = -5.40
            collision = -1.10


class GR2ReachPlantedCurriculumZ21CfgPPO(
    GR2ReachPlantedCurriculumZ17CfgPPO,
    GR2ReachPlantedCurriculumZ21Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ17CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z21_compact_midlow_from_z17_15599"
        max_iterations = 300
        save_interval = 50

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ17CfgPPO.algorithm):
        learning_rate = 7.0e-7
        learning_rate_min = 3.5e-7
        learning_rate_max = 2.8e-6
        desired_kl = 0.0022

    class policy(GR2ReachPlantedCurriculumZ17CfgPPO.policy):
        init_noise_std = [0.012] * GR2ReachPlantedCurriculumZ21Cfg.env.num_actions
