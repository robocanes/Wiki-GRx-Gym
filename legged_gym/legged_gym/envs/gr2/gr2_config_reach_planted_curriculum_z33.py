from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z32 import (
    GR2ReachPlantedCurriculumZ32Cfg,
    GR2ReachPlantedCurriculumZ32CfgPPO,
)


class GR2ReachPlantedCurriculumZ33Cfg(GR2ReachPlantedCurriculumZ32Cfg):
    """Guarded mid-low posture search with a little more exploration."""

    class rewards(GR2ReachPlantedCurriculumZ32Cfg.rewards):
        pickup_terminate_foot_disp = 0.14
        pickup_terminate_midpoint_drift = 0.060
        pickup_terminate_feet_width = 0.36

        pickup_base_pitch_min = 0.025
        pickup_base_pitch_target = 0.24

        class scales(GR2ReachPlantedCurriculumZ32Cfg.rewards.scales):
            reach_target_pos = 1.50
            reach_target_success = 0.15
            reach_target_stable = 0.70

            low_target_forward_torso_pitch = 34.00
            low_target_base_forward_pitch = 30.00
            low_target_com_forward = 26.00
            low_target_com_support = -90.00

            low_target_excess_knee_flexion = -20.00
            low_target_excess_base_height_drop = -42.00
            low_target_lower_body_ground_clearance = -48.00
            low_target_stance_width = -30.00
            low_target_abs_stance_width = -135.00
            low_target_feet_midpoint_drift = -145.00

            feet_xy_displacement = -30.00
            feet_speed_xy = -2.80
            inactive_arm_still = -0.75


class GR2ReachPlantedCurriculumZ33CfgPPO(
    GR2ReachPlantedCurriculumZ32CfgPPO,
    GR2ReachPlantedCurriculumZ33Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ32CfgPPO.runner):
        run_name = "planted_curriculum_z33_guarded_pitch_search_from_z5_13746"
        max_iterations = 140
        save_interval = 10

    class policy(GR2ReachPlantedCurriculumZ32CfgPPO.policy):
        init_noise_std = [0.018] * GR2ReachPlantedCurriculumZ33Cfg.env.num_actions
        reset_loaded_std = True
