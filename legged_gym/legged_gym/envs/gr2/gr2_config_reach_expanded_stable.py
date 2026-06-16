from legged_gym.envs.gr2.gr2_config_reach import GR2ReachCfg, GR2ReachCfgPPO


class GR2ReachExpandedStableCfg(GR2ReachCfg):
    """Expanded reach target volume, initialized from the robot-tested head-look reach policy."""

    class env(GR2ReachCfg.env):
        episode_length_s = 6
        num_envs = 8192

    class reach(GR2ReachCfg.reach):
        # Keep the first expansion modest: enough to learn higher/lower/farther reaches,
        # but not so much that it forgets the robot-safe behavior.
        target_x_range = [0.16, 0.44]
        target_y_abs_range = [0.10, 0.30]
        target_z_range = [-0.25, 0.48]
        target_roll_range = [-0.20, 0.20]
        target_pitch_range = [-0.20, 0.20]
        target_yaw_range = [-0.35, 0.35]
        draw_target_volume_box = True

    class rewards(GR2ReachCfg.rewards):
        reach_pos_tracking_sigma = 0.09
        reach_success_distance = 0.060
        reach_stable_lin_vel_sigma = 0.14
        reach_stable_ang_vel_sigma = 0.35

        class scales(GR2ReachCfg.rewards.scales):
            reach_target_pos = 6.80
            reach_target_orient = 0.08
            reach_target_success = 1.15
            reach_target_stable = 3.40
            active_end_effector_still = -0.45
            head_look_at_target = 0.80

            # Stronger anti-drift terms for the physical robot handoff:
            # reaching should not become a slow recovery walk.
            base_still = -4.80
            main_body_dof_pos = -1.70
            inactive_arm_still = -0.08

            base_height_offset_range = 1.45
            base_flat_orient = 2.60
            torso_flat_orient = 2.70

            action_diff = -0.55
            action_diff_diff = -0.14
            dof_acc = -0.035
            dof_tor = -0.035
            termination = -10.00

            limits_dof_pos_without_ankle = -4.50
            limits_dof_vel_without_ankle = -1.20
            limits_dof_tor = -0.30


class GR2ReachExpandedStableCfgPPO(GR2ReachCfgPPO, GR2ReachExpandedStableCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2ReachCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "expanded_stable_headlook_from_robot_good"
        max_iterations = 6000
        save_interval = 100

        resume = True
        load_run = "May23_21-46-37_front_random_wrist_target_headlook_nojitter_highlowz"
        checkpoint = 4999

    class algorithm(GR2ReachCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 3.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 2.e-4
        schedule = "adaptive"
        desired_kl = 0.02

        storage_class = "RolloutStorage"

    class policy(GR2ReachCfgPPO.policy):
        # Continue from the good policy with lower exploration than a fresh reach run.
        init_noise_std = [0.12] * GR2ReachExpandedStableCfg.env.num_actions
