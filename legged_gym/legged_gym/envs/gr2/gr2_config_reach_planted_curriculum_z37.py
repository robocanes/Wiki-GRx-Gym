from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z36 import (
    GR2ReachPlantedCurriculumZ36Cfg,
    GR2ReachPlantedCurriculumZ36CfgPPO,
)


class GR2ReachPlantedCurriculumZ37Cfg(GR2ReachPlantedCurriculumZ36Cfg):
    """Bring reach pressure back onto the safer mirrored hinge scaffold."""

    class reach(GR2ReachPlantedCurriculumZ36Cfg.reach):
        target_z_range = [-0.20, 0.36]
        target_y_abs_range = [0.04, 0.16]
        target_x_range = [0.24, 0.42]

        target_edge_sample_prob = 0.08
        target_edge_high_z_weight = 0.10
        target_edge_forward_x_weight = 0.12

    class rewards(GR2ReachPlantedCurriculumZ36Cfg.rewards):
        pickup_low_target_start_z = 0.10
        pickup_low_target_full_z = -0.16

        class scales(GR2ReachPlantedCurriculumZ36Cfg.rewards.scales):
            reach_target_pos = 6.00
            reach_target_success = 1.20
            reach_target_stable = 1.60
            active_end_effector_still = -0.25

            low_target_torso_pitch = 7.00
            low_target_forward_torso_pitch = 16.00
            low_target_base_forward_pitch = 16.00
            low_target_com_forward = 16.00
            low_target_com_support = -48.00

            low_target_abs_stance_width = -90.00
            low_target_feet_midpoint_drift = -95.00
            feet_xy_displacement = -18.00
            inactive_arm_still = -0.70
            main_body_dof_pos = -0.30


class GR2ReachPlantedCurriculumZ37CfgPPO(
    GR2ReachPlantedCurriculumZ36CfgPPO,
    GR2ReachPlantedCurriculumZ37Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ36CfgPPO.runner):
        run_name = "planted_curriculum_z37_reach_recovery_from_hinge_scaffold_z5_13746"
        max_iterations = 160
        save_interval = 10

    class policy(GR2ReachPlantedCurriculumZ36CfgPPO.policy):
        init_noise_std = [0.014] * GR2ReachPlantedCurriculumZ37Cfg.env.num_actions
        reset_loaded_std = True
