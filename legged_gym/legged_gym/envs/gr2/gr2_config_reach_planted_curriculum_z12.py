from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z10 import (
    GR2ReachPlantedCurriculumZ10Cfg,
    GR2ReachPlantedCurriculumZ10CfgPPO,
)


class GR2ReachPlantedCurriculumZ12Cfg(GR2ReachPlantedCurriculumZ10Cfg):
    """Keep Z10 reach while discouraging arm-leg body collisions."""

    class asset(GR2ReachPlantedCurriculumZ10Cfg.asset):
        penalize_contacts_on = [
            GR2ReachPlantedCurriculumZ10Cfg.asset.upper_arm_name,
            GR2ReachPlantedCurriculumZ10Cfg.asset.lower_arm_name,
            GR2ReachPlantedCurriculumZ10Cfg.asset.hand_name,
            GR2ReachPlantedCurriculumZ10Cfg.asset.end_effector_name,
        ]

    class rewards(GR2ReachPlantedCurriculumZ10Cfg.rewards):
        arm_leg_clearance_margin = 0.15
        pickup_feet_xy_deadband = 0.020
        pickup_feet_spread_allowance = 0.035
        pickup_feet_width_increase_target = 0.050

        class scales(GR2ReachPlantedCurriculumZ10Cfg.rewards.scales):
            reach_target_pos = 38.00
            reach_target_success = 20.00
            reach_target_stable = 14.00
            head_look_at_target = 1.80

            collision = -0.45
            arm_leg_clearance = -3.00

            base_still = -3.55
            base_xy_displacement = -3.75
            feet_xy_displacement = -2.95
            feet_speed_xy = -0.90
            inactive_arm_still = -0.17

            low_target_knee_flexion = 2.50
            low_target_hip_flexion = 2.50
            low_target_base_height_drop = 1.80
            low_target_ankle_pitch = 0.75
            low_target_yaw_twist = -6.50
            low_target_squat_coherence = 0.70
            low_target_leg_asymmetry = -3.30
            low_target_active_shoulder_drop = 5.00
            low_target_torso_pitch = 0.00
            low_target_forward_torso_pitch = 4.00
            low_target_torso_lateral_tilt = -4.00
            low_target_stance_width = -0.70

            base_height_offset_range = 0.10
            base_flat_orient = 0.35
            torso_flat_orient = 0.18

            main_body_dof_pos = -0.20
            action_diff = -0.50
            action_diff_diff = -0.12


class GR2ReachPlantedCurriculumZ12CfgPPO(
    GR2ReachPlantedCurriculumZ10CfgPPO,
    GR2ReachPlantedCurriculumZ12Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ10CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z12_arm_leg_clearance_from_z10_15000"
        max_iterations = 700
        save_interval = 50

        resume = True
        load_run = "Jun18_16-13-25_planted_curriculum_z10_reach_reopen_from_z9_14450_24576env"
        checkpoint = 15000

    class algorithm(GR2ReachPlantedCurriculumZ10CfgPPO.algorithm):
        learning_rate = 1.5e-6
        learning_rate_min = 8.0e-7
        learning_rate_max = 6.0e-6
        desired_kl = 0.004

    class policy(GR2ReachPlantedCurriculumZ10CfgPPO.policy):
        init_noise_std = [0.020] * GR2ReachPlantedCurriculumZ12Cfg.env.num_actions
