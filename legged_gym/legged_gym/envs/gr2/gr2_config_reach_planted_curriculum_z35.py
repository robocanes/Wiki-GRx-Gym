from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z34 import (
    GR2ReachPlantedCurriculumZ34Cfg,
    GR2ReachPlantedCurriculumZ34CfgPPO,
)


class GR2ReachPlantedCurriculumZ35Cfg(GR2ReachPlantedCurriculumZ34Cfg):
    """Forward-hinge pose scaffold with support-hull rewards."""

    class init_state(GR2ReachPlantedCurriculumZ34Cfg.init_state):
        default_joint_angles = GR2ReachPlantedCurriculumZ34Cfg.init_state.default_joint_angles.copy()
        default_joint_angles.update({
            "left_hip_pitch_joint": -0.38,
            "right_hip_pitch_joint": -0.38,
            "left_knee_pitch_joint": 0.34,
            "right_knee_pitch_joint": 0.34,
            "left_ankle_pitch_joint": -0.02,
            "right_ankle_pitch_joint": -0.02,
        })

    class rewards(GR2ReachPlantedCurriculumZ34Cfg.rewards):
        pickup_knee_flexion_target = 0.08
        pickup_knee_flexion_allowance = 0.040
        pickup_base_height_drop_target = 0.020
        pickup_base_height_drop_allowance = 0.008

        pickup_torso_pitch_min = 0.015
        pickup_torso_pitch_target = 0.24
        pickup_base_pitch_min = 0.015
        pickup_base_pitch_target = 0.18

        class scales(GR2ReachPlantedCurriculumZ34Cfg.rewards.scales):
            reach_target_pos = 1.00
            reach_target_success = 0.08
            reach_target_stable = 0.55

            low_target_hip_flexion = 0.00
            low_target_torso_pitch = 12.00
            low_target_forward_torso_pitch = 32.00
            low_target_base_forward_pitch = 32.00
            low_target_com_forward = 30.00
            low_target_com_support = -100.00

            low_target_excess_knee_flexion = -48.00
            low_target_excess_base_height_drop = -90.00
            main_body_dof_pos = -0.35


class GR2ReachPlantedCurriculumZ35CfgPPO(
    GR2ReachPlantedCurriculumZ34CfgPPO,
    GR2ReachPlantedCurriculumZ35Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ34CfgPPO.runner):
        run_name = "planted_curriculum_z35_forward_hinge_scaffold_from_z5_13746"
        max_iterations = 100
        save_interval = 10

    class policy(GR2ReachPlantedCurriculumZ34CfgPPO.policy):
        init_noise_std = [0.010] * GR2ReachPlantedCurriculumZ35Cfg.env.num_actions
        reset_loaded_std = True
