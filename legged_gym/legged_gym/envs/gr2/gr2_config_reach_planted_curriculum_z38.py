from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z37 import (
    GR2ReachPlantedCurriculumZ37Cfg,
    GR2ReachPlantedCurriculumZ37CfgPPO,
)


class GR2ReachPlantedCurriculumZ38Cfg(GR2ReachPlantedCurriculumZ37Cfg):
    """Reach recovery with tighter lateral symmetry and less side target pressure."""

    class reach(GR2ReachPlantedCurriculumZ37Cfg.reach):
        target_y_abs_range = [0.02, 0.10]
        target_x_range = [0.26, 0.44]

        target_edge_outer_y_weight = 0.00
        target_edge_forward_x_weight = 0.16

    class rewards(GR2ReachPlantedCurriculumZ37Cfg.rewards):
        class scales(GR2ReachPlantedCurriculumZ37Cfg.rewards.scales):
            reach_target_pos = 6.80
            reach_target_success = 1.50
            reach_target_stable = 1.80

            low_target_torso_lateral_tilt = -34.00
            low_target_leg_asymmetry = -28.00
            low_target_yaw_twist = -26.00
            low_target_com_support = -56.00
            low_target_feet_midpoint_drift = -105.00
            feet_xy_displacement = -20.00


class GR2ReachPlantedCurriculumZ38CfgPPO(
    GR2ReachPlantedCurriculumZ37CfgPPO,
    GR2ReachPlantedCurriculumZ38Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ37CfgPPO.runner):
        run_name = "planted_curriculum_z38_forward_reach_symmetric_from_hinge_z5_13746"
        max_iterations = 160
        save_interval = 10
