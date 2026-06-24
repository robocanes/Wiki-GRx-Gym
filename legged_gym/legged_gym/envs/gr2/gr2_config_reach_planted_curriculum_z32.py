from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z29 import (
    GR2ReachPlantedCurriculumZ29Cfg,
    GR2ReachPlantedCurriculumZ29CfgPPO,
)


class GR2ReachPlantedCurriculumZ32Cfg(GR2ReachPlantedCurriculumZ29Cfg):
    """Mid-low posture rehearsal with foot-escape termination."""

    class reach(GR2ReachPlantedCurriculumZ29Cfg.reach):
        target_z_range = [-0.12, 0.22]
        target_y_abs_range = [0.02, 0.08]
        target_x_range = [0.18, 0.28]

        target_edge_sample_prob = 0.02
        target_edge_low_z_weight = 0.00
        target_edge_high_z_weight = 0.05
        target_edge_outer_y_weight = 0.00
        target_edge_forward_x_weight = 0.06

    class rewards(GR2ReachPlantedCurriculumZ29Cfg.rewards):
        pickup_low_target_start_z = 0.18
        pickup_low_target_full_z = -0.08

        pickup_feet_xy_deadband = 0.010
        pickup_feet_spread_allowance = 0.002
        pickup_feet_width_max = 0.285
        pickup_feet_midpoint_deadband = 0.008

        pickup_com_forward_min = 0.015
        pickup_com_forward_target = 0.09
        pickup_torso_pitch_min = 0.04
        pickup_torso_pitch_target = 0.34
        pickup_knee_flexion_target = 0.12
        pickup_base_height_drop_target = 0.035
        pickup_base_height_drop_allowance = 0.010
        pickup_terminate_on_foot_escape = True
        pickup_terminate_low_target_gate = 0.20
        pickup_terminate_foot_disp = 0.12
        pickup_terminate_midpoint_drift = 0.050
        pickup_terminate_feet_width = 0.34

        class scales(GR2ReachPlantedCurriculumZ29Cfg.rewards.scales):
            reach_target_pos = 1.80
            reach_target_success = 0.20
            reach_target_stable = 0.80

            low_target_forward_torso_pitch = 46.00
            low_target_com_forward = 22.00
            low_target_com_support = -80.00

            low_target_excess_knee_flexion = -24.00
            low_target_excess_base_height_drop = -52.00
            low_target_lower_body_ground_clearance = -48.00
            low_target_stance_width = -36.00
            low_target_abs_stance_width = -180.00
            low_target_feet_midpoint_drift = -180.00

            feet_xy_displacement = -36.00
            feet_speed_xy = -3.20
            inactive_arm_still = -0.85


class GR2ReachPlantedCurriculumZ32CfgPPO(
    GR2ReachPlantedCurriculumZ29CfgPPO,
    GR2ReachPlantedCurriculumZ32Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ29CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z32_midlow_posture_foot_guard_from_z5_13746"
        max_iterations = 120
        save_interval = 10

        resume = True
        load_run = "Jun17_16-09-49_planted_curriculum_z5_low_squat_from_z4_13397"
        checkpoint = 13746

    class policy(GR2ReachPlantedCurriculumZ29CfgPPO.policy):
        init_noise_std = [0.006] * GR2ReachPlantedCurriculumZ32Cfg.env.num_actions
        reset_loaded_std = True
