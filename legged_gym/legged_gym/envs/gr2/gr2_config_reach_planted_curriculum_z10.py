from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z9 import (
    GR2ReachPlantedCurriculumZ9Cfg,
    GR2ReachPlantedCurriculumZ9CfgPPO,
)


class GR2ReachPlantedCurriculumZ10Cfg(GR2ReachPlantedCurriculumZ9Cfg):
    """Recover reach from Z9's cleaner 14450 posture checkpoint."""

    class reach(GR2ReachPlantedCurriculumZ9Cfg.reach):
        target_x_range = [0.10, 0.32]
        target_y_abs_range = [0.08, 0.25]
        target_z_range = [-0.52, 0.00]

        target_edge_sample_prob = 0.94
        target_edge_z_band_fraction = 0.38
        target_edge_y_band_fraction = 0.22
        target_edge_x_band_fraction = 0.18
        target_edge_low_z_weight = 0.90
        target_edge_high_z_weight = 0.01
        target_edge_outer_y_weight = 0.08
        target_edge_forward_x_weight = 0.08

    class rewards(GR2ReachPlantedCurriculumZ9Cfg.rewards):
        pickup_shoulder_drop_target = 0.09
        pickup_torso_pitch_min = 0.02
        pickup_torso_pitch_target = 0.20

        class scales(GR2ReachPlantedCurriculumZ9Cfg.rewards.scales):
            reach_target_pos = 38.00
            reach_target_success = 20.00
            reach_target_stable = 14.00
            head_look_at_target = 1.80

            base_still = -3.50
            base_xy_displacement = -3.70
            feet_xy_displacement = -2.75
            feet_speed_xy = -0.85
            inactive_arm_still = -0.15

            low_target_knee_flexion = 2.60
            low_target_hip_flexion = 2.60
            low_target_base_height_drop = 1.85
            low_target_ankle_pitch = 0.80
            low_target_yaw_twist = -6.20
            low_target_squat_coherence = 0.75
            low_target_leg_asymmetry = -3.20
            low_target_active_shoulder_drop = 5.00
            low_target_torso_pitch = 0.00
            low_target_forward_torso_pitch = 6.20
            low_target_torso_lateral_tilt = -3.80
            low_target_stance_width = 0.00

            base_height_offset_range = 0.10
            base_flat_orient = 0.35
            torso_flat_orient = 0.18

            main_body_dof_pos = -0.20
            action_diff = -0.50
            action_diff_diff = -0.12


class GR2ReachPlantedCurriculumZ10CfgPPO(
    GR2ReachPlantedCurriculumZ9CfgPPO,
    GR2ReachPlantedCurriculumZ10Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ9CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z10_reach_reopen_from_z9_14450"
        max_iterations = 1200
        save_interval = 50

        resume = True
        load_run = "Jun18_15-42-50_planted_curriculum_z9_reach_rebalance_from_z8_14350_24576env"
        checkpoint = 14450

    class algorithm(GR2ReachPlantedCurriculumZ9CfgPPO.algorithm):
        learning_rate = 2.0e-6
        learning_rate_min = 1.0e-6
        learning_rate_max = 8.0e-6
        desired_kl = 0.005

    class policy(GR2ReachPlantedCurriculumZ9CfgPPO.policy):
        init_noise_std = [0.025] * GR2ReachPlantedCurriculumZ10Cfg.env.num_actions
