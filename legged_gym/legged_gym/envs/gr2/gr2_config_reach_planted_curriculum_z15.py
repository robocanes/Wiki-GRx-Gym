from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z14 import (
    GR2ReachPlantedCurriculumZ14Cfg,
    GR2ReachPlantedCurriculumZ14CfgPPO,
)


class GR2ReachPlantedCurriculumZ15Cfg(GR2ReachPlantedCurriculumZ14Cfg):
    """Continue from Z14 before drift, with direct thigh/shank contact pressure."""

    class rewards(GR2ReachPlantedCurriculumZ14Cfg.rewards):
        pickup_lower_body_ground_clearance_min = 0.24
        pickup_lower_body_contact_force_threshold = 0.10
        pickup_base_height_drop_allowance = 0.025
        pickup_feet_width_increase_target = 0.038

        class scales(GR2ReachPlantedCurriculumZ14Cfg.rewards.scales):
            low_target_lower_body_contact = -4.00
            low_target_lower_body_ground_clearance = -12.00
            low_target_excess_base_height_drop = -12.00
            low_target_stance_width = -1.45
            low_target_leg_asymmetry = -4.40

            low_target_knee_flexion = 1.50
            low_target_hip_flexion = 1.60
            low_target_base_height_drop = 0.90
            feet_xy_displacement = -3.40
            inactive_arm_still = -0.22
            arm_leg_clearance = -0.75


class GR2ReachPlantedCurriculumZ15CfgPPO(
    GR2ReachPlantedCurriculumZ14CfgPPO,
    GR2ReachPlantedCurriculumZ15Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ14CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z15_lower_body_contact_from_z14_15100"
        max_iterations = 550
        save_interval = 50

        resume = True
        load_run = "Jun18_21-43-34_planted_curriculum_z14_anti_kneel_from_z10_15000_24576env"
        checkpoint = 15100

    class algorithm(GR2ReachPlantedCurriculumZ14CfgPPO.algorithm):
        learning_rate = 1.2e-6
        learning_rate_min = 6.0e-7
        learning_rate_max = 5.0e-6
        desired_kl = 0.0035

    class policy(GR2ReachPlantedCurriculumZ14CfgPPO.policy):
        init_noise_std = [0.018] * GR2ReachPlantedCurriculumZ15Cfg.env.num_actions
