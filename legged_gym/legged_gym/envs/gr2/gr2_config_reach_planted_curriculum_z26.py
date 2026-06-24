from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z24 import (
    GR2ReachPlantedCurriculumZ24Cfg,
    GR2ReachPlantedCurriculumZ24CfgPPO,
)


class GR2ReachPlantedCurriculumZ26Cfg(GR2ReachPlantedCurriculumZ24Cfg):
    """Stance-first branch from the visually better Z10 actor mean."""

    class reach(GR2ReachPlantedCurriculumZ24Cfg.reach):
        target_z_range = [-0.42, 0.10]
        target_y_abs_range = [0.03, 0.14]
        target_x_range = [0.18, 0.32]

        target_edge_sample_prob = 0.20
        target_edge_low_z_weight = 0.06
        target_edge_high_z_weight = 0.16
        target_edge_outer_y_weight = 0.03
        target_edge_forward_x_weight = 0.10

    class rewards(GR2ReachPlantedCurriculumZ24Cfg.rewards):
        pickup_low_target_start_z = -0.22
        pickup_low_target_full_z = -0.40

        class scales(GR2ReachPlantedCurriculumZ24Cfg.rewards.scales):
            reach_target_pos = 12.00
            reach_target_success = 4.00
            reach_target_stable = 3.00

            low_target_abs_stance_width = -100.00
            low_target_feet_midpoint_drift = -90.00
            low_target_stance_width = -16.00
            feet_xy_displacement = -14.00
            inactive_arm_still = -0.45


class GR2ReachPlantedCurriculumZ26CfgPPO(
    GR2ReachPlantedCurriculumZ24CfgPPO,
    GR2ReachPlantedCurriculumZ26Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ24CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z26_stance_from_z10_15000"
        max_iterations = 160
        save_interval = 10

        resume = True
        load_run = "Jun18_16-13-25_planted_curriculum_z10_reach_reopen_from_z9_14450_24576env"
        checkpoint = 15000

    class algorithm(GR2ReachPlantedCurriculumZ24CfgPPO.algorithm):
        learning_rate = 5.0e-7
        learning_rate_min = 2.5e-7
        learning_rate_max = 2.0e-6
        desired_kl = 0.0018

    class policy(GR2ReachPlantedCurriculumZ24CfgPPO.policy):
        init_noise_std = [0.010] * GR2ReachPlantedCurriculumZ26Cfg.env.num_actions
        reset_loaded_std = True
