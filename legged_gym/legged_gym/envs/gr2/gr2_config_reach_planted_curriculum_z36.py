from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z35 import (
    GR2ReachPlantedCurriculumZ35Cfg,
    GR2ReachPlantedCurriculumZ35CfgPPO,
)


class GR2ReachPlantedCurriculumZ36Cfg(GR2ReachPlantedCurriculumZ35Cfg):
    """Mirror the hip-hinge scaffold to test the forward pitch sign convention."""

    class init_state(GR2ReachPlantedCurriculumZ35Cfg.init_state):
        default_joint_angles = GR2ReachPlantedCurriculumZ35Cfg.init_state.default_joint_angles.copy()
        default_joint_angles.update({
            "left_hip_pitch_joint": 0.22,
            "right_hip_pitch_joint": 0.22,
            "left_knee_pitch_joint": 0.30,
            "right_knee_pitch_joint": 0.30,
            "left_ankle_pitch_joint": -0.10,
            "right_ankle_pitch_joint": -0.10,
        })


class GR2ReachPlantedCurriculumZ36CfgPPO(
    GR2ReachPlantedCurriculumZ35CfgPPO,
    GR2ReachPlantedCurriculumZ36Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ35CfgPPO.runner):
        run_name = "planted_curriculum_z36_forward_hinge_mirror_from_z5_13746"
        max_iterations = 80
        save_interval = 10
