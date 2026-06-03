from legged_gym.envs.gr2.gr2_config_upper_body import GR2UpperBodyCfg, GR2UpperBodyCfgPPO


class GR2DynamicWalkCfg(GR2UpperBodyCfg):
    class terrain(GR2UpperBodyCfg.terrain):
        mesh_type = "trimesh"
        curriculum = False
        num_rows = 2
        num_cols = 4
        max_init_terrain_level = 0

        terrain_proportions = [
            0.90, 0.10,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]

    class commands(GR2UpperBodyCfg.commands):
        curriculum = False
        command_profile = "base_velocity"
        resample_command_interval_s = 6.0
        gait_patterns = [
            "stand", "walk",
        ]

        class ranges(GR2UpperBodyCfg.commands.ranges):
            lin_vel_x = [0.10, 0.85]
            lin_vel_y = [-0.18, 0.18]
            ang_vel_yaw = [-0.35, 0.35]

    class control(GR2UpperBodyCfg.control):
        action_scale = {
            **GR2UpperBodyCfg.control.action_scale,

            "waist_yaw": 0.55,

            "shoulder_pitch": 0.34,
            "shoulder_roll": 0.04,
            "shoulder_yaw": 0.05,
            "elbow_pitch": 0.14,
        }

    class rewards(GR2UpperBodyCfg.rewards):
        gait_cycle_period = 0.70
        feet_air_time_target = 0.30
        base_height_target = 0.99
        base_height_offset_range_limit = 0.015

        class scales(GR2UpperBodyCfg.rewards.scales):
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
            shoulder_pitch_pos = -0.08
            shoulder_pitch_vel = -0.012
            shoulder_pitch_swing = 0.32
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


class GR2DynamicWalkCfgPPO(GR2UpperBodyCfgPPO, GR2DynamicWalkCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2UpperBodyCfgPPO.runner):
        experiment_name = "GR2UpperBody"
        num_steps_per_env = 64

        run_name = "dynamic_walk_expressive_shoulder_pitch_swing"
        max_iterations = 2000
        save_interval = 100

        resume = True
        load_run = "May31_12-45-15_dynamic_walk_more_forward_arm_swing"
        checkpoint = 15998

    class algorithm(GR2UpperBodyCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 25
        learning_rate = 5.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 5.e-4
        schedule = "adaptive"
        desired_kl = 0.025

        storage_class = "RolloutStorage"

    class policy(GR2UpperBodyCfgPPO.policy):
        init_noise_std = [0.28] * GR2DynamicWalkCfg.env.num_actions
