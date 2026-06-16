from legged_gym.envs.gr2.gr2_config_reach_planted_pickup import (
    GR2ReachPlantedPickup,
    GR2ReachPlantedPickupCfg,
    GR2ReachPlantedPickupCfgPPO,
)


class GR2ReachPlantedCurriculumZ1Cfg(GR2ReachPlantedPickupCfg):
    """First planted-pickup curriculum stage: modest low-z expansion, no hard floor targets yet."""

    class reach(GR2ReachPlantedPickupCfg.reach):
        target_x_range = [0.16, 0.42]
        target_y_abs_range = [0.10, 0.32]
        target_z_range = [-0.20, 0.55]

    class rewards(GR2ReachPlantedPickupCfg.rewards):
        pickup_low_target_start_z = 0.02
        pickup_low_target_full_z = -0.18
        pickup_base_xy_deadband = 0.050
        pickup_feet_xy_deadband = 0.040
        pickup_knee_flexion_target = 0.32
        pickup_hip_flexion_target = 0.16

        class scales(GR2ReachPlantedPickupCfg.rewards.scales):
            reach_target_pos = 7.10
            reach_target_orient = 0.06
            reach_target_success = 1.20
            reach_target_stable = 3.25
            active_end_effector_still = -0.36
            head_look_at_target = 0.85

            # Stage 1: nudge planted behavior without making the existing policy die.
            base_still = -2.90
            base_xy_displacement = -1.20
            feet_xy_displacement = -0.85
            feet_speed_xy = -0.22
            low_target_knee_flexion = 5.50
            low_target_hip_flexion = 2.00

            main_body_dof_pos = -0.25
            inactive_arm_still = -0.06

            base_height_offset_range = 0.35
            base_flat_orient = 1.75
            torso_flat_orient = 1.95

            action_diff = -0.45
            action_diff_diff = -0.10
            dof_acc = -0.030
            dof_tor = -0.030
            termination = -10.00

            limits_dof_pos_without_ankle = -4.50
            limits_dof_vel_without_ankle = -1.20
            limits_dof_tor = -0.30


class GR2ReachPlantedCurriculumZ1CfgPPO(
    GR2ReachPlantedPickupCfgPPO,
    GR2ReachPlantedCurriculumZ1Cfg,
):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2ReachPlantedPickupCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z1_from_expanded_stable"
        max_iterations = 1000
        save_interval = 100

        resume = True
        load_run = "Jun15_22-35-46_expanded_stable_headlook_from_robot_good"
        checkpoint = 10998

    class algorithm(GR2ReachPlantedPickupCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 1.5e-5
        learning_rate_min = 5.e-6
        learning_rate_max = 1.0e-4
        schedule = "adaptive"
        desired_kl = 0.015

        storage_class = "RolloutStorage"

    class policy(GR2ReachPlantedPickupCfgPPO.policy):
        init_noise_std = [0.08] * GR2ReachPlantedCurriculumZ1Cfg.env.num_actions
