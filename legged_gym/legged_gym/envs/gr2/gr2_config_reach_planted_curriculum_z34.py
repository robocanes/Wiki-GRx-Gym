from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z33 import (
    GR2ReachPlantedCurriculumZ33Cfg,
    GR2ReachPlantedCurriculumZ33CfgPPO,
)


class GR2ReachPlantedCurriculumZ34Cfg(GR2ReachPlantedCurriculumZ33Cfg):
    """Hip-hinge posture rehearsal: forward body over planted feet, not a deep sit."""

    class rewards(GR2ReachPlantedCurriculumZ33Cfg.rewards):
        pickup_hip_flexion_target = 0.30
        pickup_knee_flexion_target = 0.10
        pickup_knee_flexion_allowance = 0.045
        pickup_base_height_drop_target = 0.025
        pickup_base_height_drop_allowance = 0.008

        pickup_torso_pitch_min = 0.025
        pickup_torso_pitch_target = 0.30
        pickup_base_pitch_min = 0.020
        pickup_base_pitch_target = 0.20

        class scales(GR2ReachPlantedCurriculumZ33Cfg.rewards.scales):
            reach_target_pos = 1.20
            reach_target_success = 0.10
            reach_target_stable = 0.60

            low_target_hip_flexion = 26.00
            low_target_knee_flexion = 0.00
            low_target_excess_knee_flexion = -44.00
            low_target_base_height_drop = 0.00
            low_target_excess_base_height_drop = -84.00

            low_target_torso_pitch = 8.00
            low_target_forward_torso_pitch = 26.00
            low_target_base_forward_pitch = 26.00
            low_target_com_forward = 28.00
            low_target_com_support = -96.00

            low_target_lower_body_ground_clearance = -56.00
            low_target_abs_stance_width = -145.00
            low_target_feet_midpoint_drift = -155.00
            feet_xy_displacement = -32.00


class GR2ReachPlantedCurriculumZ34CfgPPO(
    GR2ReachPlantedCurriculumZ33CfgPPO,
    GR2ReachPlantedCurriculumZ34Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ33CfgPPO.runner):
        run_name = "planted_curriculum_z34_hip_hinge_from_z5_13746"
        max_iterations = 140
        save_interval = 10

    class policy(GR2ReachPlantedCurriculumZ33CfgPPO.policy):
        init_noise_std = [0.014] * GR2ReachPlantedCurriculumZ34Cfg.env.num_actions
        reset_loaded_std = True
