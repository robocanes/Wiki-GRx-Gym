from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z32 import (
    GR2ReachPlantedCurriculumZ32Cfg,
    GR2ReachPlantedCurriculumZ32CfgPPO,
)


class GR2ReachPlantedCurriculumZ32WideCfg(GR2ReachPlantedCurriculumZ32Cfg):
    """Demo/play variant of stable Z32 with a wider but still conservative target box."""

    class reach(GR2ReachPlantedCurriculumZ32Cfg.reach):
        target_x_range = [0.14, 0.36]
        target_y_abs_range = [0.02, 0.16]
        target_z_range = [-0.14, 0.34]

        target_edge_sample_prob = 0.08
        target_edge_low_z_weight = 0.02
        target_edge_high_z_weight = 0.12
        target_edge_outer_y_weight = 0.04
        target_edge_forward_x_weight = 0.12


class GR2ReachPlantedCurriculumZ32WideCfgPPO(
    GR2ReachPlantedCurriculumZ32CfgPPO,
    GR2ReachPlantedCurriculumZ32WideCfg,
):
    class runner(GR2ReachPlantedCurriculumZ32CfgPPO.runner):
        run_name = "planted_curriculum_z32_wide_demo_targets"
