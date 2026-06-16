from legged_gym.envs.gr2.gr2_config_reach_expanded_stable import (
    GR2ReachExpandedStableCfg,
    GR2ReachExpandedStableCfgPPO,
)


class GR2ReachPickupHighCfg(GR2ReachExpandedStableCfg):
    """Second-stage reach: lower floor-pick targets, higher overhead targets, same forward bound."""

    class reach(GR2ReachExpandedStableCfg.reach):
        # Keep forward reach no farther than the tested expanded policy.
        target_x_range = [0.16, 0.44]
        # Slightly wider side range only.
        target_y_abs_range = [0.10, 0.34]
        # Add floor-pick and higher-overhead coverage.
        target_z_range = [-0.40, 0.62]
        target_roll_range = [-0.20, 0.20]
        target_pitch_range = [-0.22, 0.22]
        target_yaw_range = [-0.35, 0.35]
        draw_target_volume_box = True

    class rewards(GR2ReachExpandedStableCfg.rewards):
        reach_pos_tracking_sigma = 0.10
        reach_success_distance = 0.065
        reach_stable_lin_vel_sigma = 0.14
        reach_stable_ang_vel_sigma = 0.35

        class scales(GR2ReachExpandedStableCfg.rewards.scales):
            reach_target_pos = 7.20
            reach_target_orient = 0.07
            reach_target_success = 1.25
            reach_target_stable = 3.30
            active_end_effector_still = -0.38
            head_look_at_target = 0.85

            # Still discourage drift, but let knees/hips/base height move enough for low targets.
            base_still = -4.60
            main_body_dof_pos = -1.35
            inactive_arm_still = -0.07

            base_height_offset_range = 0.80
            base_flat_orient = 2.15
            torso_flat_orient = 2.35

            action_diff = -0.50
            action_diff_diff = -0.12
            dof_acc = -0.032
            dof_tor = -0.032
            termination = -10.00

            limits_dof_pos_without_ankle = -4.50
            limits_dof_vel_without_ankle = -1.20
            limits_dof_tor = -0.30


class GR2ReachPickupHighCfgPPO(GR2ReachExpandedStableCfgPPO, GR2ReachPickupHighCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2ReachExpandedStableCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "pickup_high_from_expanded_stable"
        max_iterations = 4000
        save_interval = 100

        resume = True
        load_run = "Jun15_22-35-46_expanded_stable_headlook_from_robot_good"
        checkpoint = 10998

    class algorithm(GR2ReachExpandedStableCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 2.e-5
        learning_rate_min = 8.e-6
        learning_rate_max = 1.5e-4
        schedule = "adaptive"
        desired_kl = 0.018

        storage_class = "RolloutStorage"

    class policy(GR2ReachExpandedStableCfgPPO.policy):
        # Keep exploration modest: this is a range extension, not a new skill from scratch.
        init_noise_std = [0.10] * GR2ReachPickupHighCfg.env.num_actions
