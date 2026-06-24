from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z15 import (
    GR2ReachPlantedCurriculumZ15Cfg,
    GR2ReachPlantedCurriculumZ15CfgPPO,
)


class GR2ReachPlantedCurriculumZ16Cfg(GR2ReachPlantedCurriculumZ15Cfg):
    """Favor forward torso hinge over knee-down/seated collision strategies."""

    class reach(GR2ReachPlantedCurriculumZ15Cfg.reach):
        target_z_range = [-0.46, 0.00]
        target_edge_low_z_weight = 0.78
        target_edge_high_z_weight = 0.04

    class rewards(GR2ReachPlantedCurriculumZ15Cfg.rewards):
        pickup_torso_pitch_min = 0.08
        pickup_torso_pitch_target = 0.34
        pickup_lower_body_ground_clearance_min = 0.28
        pickup_lower_body_contact_force_threshold = 0.08
        pickup_base_height_drop_allowance = 0.018
        pickup_feet_width_increase_target = 0.034

        class scales(GR2ReachPlantedCurriculumZ15Cfg.rewards.scales):
            low_target_forward_torso_pitch = 9.00
            low_target_active_shoulder_drop = 5.20

            low_target_knee_flexion = 0.70
            low_target_hip_flexion = 0.95
            low_target_base_height_drop = 0.45
            low_target_squat_coherence = 0.25
            low_target_ankle_pitch = 0.35

            low_target_lower_body_contact = -8.00
            low_target_lower_body_ground_clearance = -24.00
            low_target_excess_base_height_drop = -18.00
            low_target_stance_width = -1.80
            low_target_leg_asymmetry = -4.80
            low_target_torso_lateral_tilt = -5.00
            low_target_yaw_twist = -7.40

            feet_xy_displacement = -3.70
            feet_speed_xy = -1.05
            inactive_arm_still = -0.24
            arm_leg_clearance = -1.00
            collision = -0.60


class GR2ReachPlantedCurriculumZ16CfgPPO(
    GR2ReachPlantedCurriculumZ15CfgPPO,
    GR2ReachPlantedCurriculumZ16Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ15CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z16_forward_hinge_from_z14_15100"
        max_iterations = 500
        save_interval = 50

        resume = True
        load_run = "Jun18_21-43-34_planted_curriculum_z14_anti_kneel_from_z10_15000_24576env"
        checkpoint = 15100

    class algorithm(GR2ReachPlantedCurriculumZ15CfgPPO.algorithm):
        learning_rate = 1.0e-6
        learning_rate_min = 5.0e-7
        learning_rate_max = 4.0e-6
        desired_kl = 0.003

    class policy(GR2ReachPlantedCurriculumZ15CfgPPO.policy):
        init_noise_std = [0.016] * GR2ReachPlantedCurriculumZ16Cfg.env.num_actions
