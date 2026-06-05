from legged_gym.envs.gr2.gr2_config_dynamic_walk import GR2DynamicWalkCfg
from legged_gym.envs.gr2.gr2_config_reach import GR2ReachCfg, GR2ReachCfgPPO


class GR2DynamicWalkReachPretrainCfg(GR2ReachCfg):
    """29-action dynamic-walk task with the GR2Reach policy shape."""

    class env(GR2ReachCfg.env):
        episode_length_s = 20
        num_envs = 8192
        num_obs = GR2ReachCfg.env.num_obs
        num_pri_obs = GR2ReachCfg.env.num_pri_obs
        num_actions = GR2ReachCfg.env.num_actions

    class terrain(GR2ReachCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class commands(GR2DynamicWalkCfg.commands):
        pass

    class control(GR2ReachCfg.control):
        action_scale = {
            **GR2ReachCfg.control.action_scale,
            "waist_yaw": 0.55,
            "head_yaw": 0.12,
            "head_pitch": 0.10,
            "shoulder_pitch": 0.22,
            "shoulder_roll": 0.04,
            "shoulder_yaw": 0.05,
            "elbow_pitch": 0.12,
            "wrist_yaw": 0.08,
            "wrist_pitch": 0.08,
            "wrist_roll": 0.08,
        }

    class rewards(GR2ReachCfg.rewards):
        gait_cycle_period = GR2DynamicWalkCfg.rewards.gait_cycle_period
        feet_air_time_target = GR2DynamicWalkCfg.rewards.feet_air_time_target
        base_height_target = GR2DynamicWalkCfg.rewards.base_height_target
        base_height_offset_range_limit = GR2DynamicWalkCfg.rewards.base_height_offset_range_limit

        only_positive_rewards = False

        class scales(GR2ReachCfg.rewards.scales):
            reach_target_pos = 0.0
            reach_target_orient = 0.0
            reach_target_success = 0.0
            reach_target_stable = 0.0
            active_end_effector_still = 0.0
            head_look_at_target = 0.0
            base_still = 0.0
            inactive_arm_still = 0.0
            main_body_dof_pos = 0.0

            stand_still_dof_pos_waist_joint = 0.40
            stand_still_foot_distance = 0.20

            cmd_diff_base_lin_vel_x = 2.70
            cmd_diff_base_lin_vel_y = 0.35
            cmd_diff_base_ang_vel_yaw = 0.55

            base_lin_vel_z = 0.35
            base_height_offset_range = 0.65
            base_flat_orient = 0.35
            torso_flat_orient = 0.45

            action_diff = -5.50
            action_diff_diff = -1.20

            dof_pos_offset = 0.28
            shoulder_pitch_pos = -0.20
            shoulder_pitch_vel = -0.025
            shoulder_sideways_pos = -1.65
            shoulder_sideways_vel = -0.10
            dof_acc = -0.24
            dof_tor = -0.05

            limits_dof_pos_without_ankle = -10.00
            limits_dof_vel_without_ankle = -5.00
            limits_dof_tor = -1.00

            feet_speed_xy_close_to_ground = 0.25
            feet_force_z_close_to_ground = -0.18
            feet_stumble = -0.25
            feet_distance_too_close = -0.55
            feet_air_time = 2.40


class GR2DynamicWalkReachPretrainCfgPPO(GR2ReachCfgPPO, GR2DynamicWalkReachPretrainCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2ReachCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 64

        run_name = "dynamic_walk_reach_shape_from_may23"
        max_iterations = 2000
        save_interval = 100

        resume = True
        load_run = "May23_21-46-37_front_random_wrist_target_headlook_nojitter_highlowz"
        checkpoint = 4999

    class algorithm(GR2ReachCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 5.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 5.e-4
        schedule = "adaptive"
        desired_kl = 0.025

        storage_class = "RolloutStorage"

    class policy(GR2ReachCfgPPO.policy):
        init_noise_std = [0.24] * GR2DynamicWalkReachPretrainCfg.env.num_actions
