from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z29 import (
    GR2ReachPlantedCurriculumZ29Cfg,
    GR2ReachPlantedCurriculumZ29CfgPPO,
)


class GR2ReachPlantedCurriculumZ30Cfg(GR2ReachPlantedCurriculumZ29Cfg):
    """Z29 rerun after fixing the forward torso-pitch sign."""


class GR2ReachPlantedCurriculumZ30CfgPPO(
    GR2ReachPlantedCurriculumZ29CfgPPO,
    GR2ReachPlantedCurriculumZ30Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ29CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z30_posture_first_signfix_from_z5_13746"
        max_iterations = 120
        save_interval = 10

        resume = True
        load_run = "Jun17_16-09-49_planted_curriculum_z5_low_squat_from_z4_13397"
        checkpoint = 13746

    class policy(GR2ReachPlantedCurriculumZ29CfgPPO.policy):
        init_noise_std = [0.008] * GR2ReachPlantedCurriculumZ30Cfg.env.num_actions
        reset_loaded_std = True
