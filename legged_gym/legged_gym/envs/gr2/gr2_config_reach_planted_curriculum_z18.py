from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z17 import (
    GR2ReachPlantedCurriculumZ17Cfg,
    GR2ReachPlantedCurriculumZ17CfgPPO,
)


class GR2ReachPlantedCurriculumZ18Cfg(GR2ReachPlantedCurriculumZ17Cfg):
    """Discourage the seated/kneeling low-target solution seen in Z17 visuals."""

    class rewards(GR2ReachPlantedCurriculumZ17Cfg.rewards):
        pickup_low_target_start_z = -0.34
        pickup_low_target_full_z = -0.58

        pickup_knee_flexion_target = 0.22
        pickup_knee_flexion_allowance = 0.08
        pickup_base_height_drop_target = 0.07
        pickup_base_height_drop_allowance = 0.025
        pickup_lower_body_ground_clearance_min = 0.34
        pickup_lower_body_contact_force_threshold = 0.06
        pickup_torso_pitch_min = 0.08
        pickup_torso_pitch_target = 0.42
        arm_leg_clearance_margin = 0.22

        class scales(GR2ReachPlantedCurriculumZ17Cfg.rewards.scales):
            reach_target_pos = 38.00
            reach_target_success = 20.00
            reach_target_stable = 14.00

            low_target_forward_torso_pitch = 8.20
            low_target_active_shoulder_drop = 3.20

            low_target_knee_flexion = -0.45
            low_target_excess_knee_flexion = -18.00
            low_target_hip_flexion = 0.20
            low_target_base_height_drop = 0.00
            low_target_squat_coherence = 0.00
            low_target_ankle_pitch = 0.10

            low_target_lower_body_contact = -16.00
            low_target_lower_body_ground_clearance = -32.00
            low_target_excess_base_height_drop = -28.00
            low_target_stance_width = -2.40
            low_target_leg_asymmetry = -7.00
            low_target_torso_lateral_tilt = -6.50
            low_target_yaw_twist = -8.50

            feet_xy_displacement = -4.30
            feet_speed_xy = -1.15
            inactive_arm_still = -0.42
            arm_leg_clearance = -4.80
            collision = -1.00


class GR2ReachPlantedCurriculumZ18CfgPPO(
    GR2ReachPlantedCurriculumZ17CfgPPO,
    GR2ReachPlantedCurriculumZ18Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ17CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z18_no_kneel_from_z17_15599"
        max_iterations = 400
        save_interval = 50

        resume = True
        load_run = "Jun18_23-49-20_planted_curriculum_z17_real_height_band_from_z14_15100_24576env"
        checkpoint = 15599

    class algorithm(GR2ReachPlantedCurriculumZ17CfgPPO.algorithm):
        learning_rate = 8.0e-7
        learning_rate_min = 4.0e-7
        learning_rate_max = 3.0e-6
        desired_kl = 0.0025

    class policy(GR2ReachPlantedCurriculumZ17CfgPPO.policy):
        init_noise_std = [0.014] * GR2ReachPlantedCurriculumZ18Cfg.env.num_actions
