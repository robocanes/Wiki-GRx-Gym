from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z13 import (
    GR2ReachPlantedCurriculumZ13Cfg,
    GR2ReachPlantedCurriculumZ13CfgPPO,
)


class GR2ReachPlantedCurriculumZ14Cfg(GR2ReachPlantedCurriculumZ13Cfg):
    """Add direct anti-kneeling pressure while preserving Z13's mild arm cleanup."""

    class rewards(GR2ReachPlantedCurriculumZ13Cfg.rewards):
        pickup_lower_body_ground_clearance_min = 0.20
        pickup_base_height_drop_allowance = 0.030
        pickup_feet_width_increase_target = 0.040

        class scales(GR2ReachPlantedCurriculumZ13Cfg.rewards.scales):
            low_target_knee_flexion = 1.75
            low_target_hip_flexion = 1.80
            low_target_base_height_drop = 1.10
            low_target_excess_base_height_drop = -10.00
            low_target_lower_body_ground_clearance = -6.00
            low_target_stance_width = -1.25
            low_target_leg_asymmetry = -4.10

            feet_xy_displacement = -3.25
            inactive_arm_still = -0.20
            arm_leg_clearance = -0.80


class GR2ReachPlantedCurriculumZ14CfgPPO(
    GR2ReachPlantedCurriculumZ13CfgPPO,
    GR2ReachPlantedCurriculumZ14Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ13CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z14_anti_kneel_from_z10_15000"
        max_iterations = 650
        save_interval = 50

        resume = True
        load_run = "Jun18_16-13-25_planted_curriculum_z10_reach_reopen_from_z9_14450_24576env"
        checkpoint = 15000

    class algorithm(GR2ReachPlantedCurriculumZ13CfgPPO.algorithm):
        learning_rate = 1.5e-6
        learning_rate_min = 8.0e-7
        learning_rate_max = 6.0e-6
        desired_kl = 0.004

    class policy(GR2ReachPlantedCurriculumZ13CfgPPO.policy):
        init_noise_std = [0.020] * GR2ReachPlantedCurriculumZ14Cfg.env.num_actions
