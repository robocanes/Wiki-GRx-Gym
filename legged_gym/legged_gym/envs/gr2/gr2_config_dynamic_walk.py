import numpy

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
            lin_vel_x = [0.05, 0.55]
            lin_vel_y = [-0.12, 0.12]
            ang_vel_yaw = [-0.35, 0.35]

    class control(GR2UpperBodyCfg.control):
        action_scale = {
            **GR2UpperBodyCfg.control.action_scale,

            "waist_yaw": 0.55,
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

            action_diff = -6.20
            action_diff_diff = -1.60

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


    class mirror(GR2UpperBodyCfg.mirror):
        enable_mirror = True
        observations_coefficient = numpy.array([
            1.0, -1.0, -1.0,
            -1.0, 1.0, -1.0,
            1.0, -1.0, 1.0,

            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            -1.0,
            -1.0, 1.0,
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,

            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            -1.0,
            -1.0, 1.0,
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,

            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            -1.0,
            1.0, -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0, 1.0,
        ])
        observations_exchange = numpy.array([
            *[(9 + i, 9 + 6 + i) for i in range(6)],
            *[(24 + i, 24 + 7 + i) for i in range(7)],
            *[(38 + i, 38 + 6 + i) for i in range(6)],
            *[(53 + i, 53 + 7 + i) for i in range(7)],
            *[(67 + i, 67 + 6 + i) for i in range(6)],
            *[(80 + i, 80 + 4 + i) for i in range(4)],
        ])
        actions_coefficient = numpy.array([
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
            -1.0,
            1.0, -1.0, -1.0, 1.0,
            1.0, -1.0, -1.0, 1.0,
        ])
        actions_exchange = numpy.array([
            *[(0 + i, 0 + 6 + i) for i in range(6)],
            *[(13 + i, 13 + 4 + i) for i in range(4)],
        ])


class GR2DynamicWalkCfgPPO(GR2UpperBodyCfgPPO, GR2DynamicWalkCfg):
    runner_class_name = "OnPolicyRunnerMirror"

    class runner(GR2UpperBodyCfgPPO.runner):
        experiment_name = "GR2UpperBody"
        num_steps_per_env = 64

        run_name = "dynamic_walk_smooth_symmetry_from_13999"
        max_iterations = 600
        save_interval = 50

        resume = True
        load_run = "May30_14-44-22_dynamic_walk_from_quiet_elbows"
        checkpoint = 13999

    class algorithm(GR2UpperBodyCfgPPO.algorithm):
        class_name = "PPOMirror"

        num_learning_epochs = 8
        num_mini_batches = 25
        learning_rate = 2.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 1.e-4
        schedule = "adaptive"
        desired_kl = 0.015
        mirror_coef = 0.15

        storage_class = "RolloutStorage"

    class policy(GR2UpperBodyCfgPPO.policy):
        init_noise_std = [0.20] * GR2DynamicWalkCfg.env.num_actions
