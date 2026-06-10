import numpy

from legged_gym.envs.gr2.gr2_config import GR2CfgPPO as GR2BaseCfgPPO
from legged_gym.envs.gr2.gr2_config_main_body import GR2MainBodyCfg


class GR2DynamicWalk12Cfg(GR2MainBodyCfg):
    class env(GR2MainBodyCfg.env):
        episode_length_s = 20

        # 49 actor observations * 40 history = 1960 actor input, matching the
        # vendor lower-body policy shape we want to replace.
        num_obs = 49
        num_pri_obs = 182
        num_actions = 12

        use_stack = True
        num_stack = 40

    class terrain(GR2MainBodyCfg.terrain):
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

    class commands(GR2MainBodyCfg.commands):
        curriculum = False
        command_profile = "base_velocity"
        resample_command_interval_s = 4.0
        yaw_spot_turn_curriculum = True
        yaw_spot_turn_fraction = 0.45
        yaw_spot_turn_positive_fraction = 0.50
        yaw_spot_turn_min_abs = 0.30
        yaw_spot_turn_max_abs = 0.55
        gait_patterns = [
            "stand", "walk",
        ]

        class ranges(GR2MainBodyCfg.commands.ranges):
            lin_vel_x = [-0.25, 0.50]
            lin_vel_y = [-0.20, 0.20]
            ang_vel_yaw = [-0.55, 0.55]

    class control(GR2MainBodyCfg.control):
        action_names = [
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_pitch",
            "ankle_roll",
        ]

        action_scale = {
            "hip_pitch": 0.90,
            "hip_roll": 0.80,
            "hip_yaw": 0.70,
            "knee_pitch": 0.90,
            "ankle_pitch": 0.80,
            "ankle_roll": 0.55,
        }

    class rewards(GR2MainBodyCfg.rewards):
        gait_cycle_period = 0.75
        feet_air_time_target = 0.32
        feet_swing_clearance_target = 0.055
        base_height_target = 0.99
        base_height_offset_range_limit = 0.015
        positive_yaw_reward_min_command = 0.15
        spot_yaw_reward_min_command = 0.15
        spot_yaw_reward_max_linear_command = 0.10

        class scales(GR2MainBodyCfg.rewards.scales):
            stand_still_dof_pos_waist_joint = 0.45
            stand_still_foot_distance = 0.20

            cmd_diff_base_lin_vel_x = 2.70
            cmd_diff_base_lin_vel_y = 0.35
            cmd_diff_base_ang_vel_yaw = 1.45
            cmd_diff_base_ang_vel_yaw_spot = 0.70

            base_lin_vel_z = 0.35
            base_height_offset_range = 0.65
            base_flat_orient = 0.35
            torso_flat_orient = 0.45

            action_diff = -6.40
            action_diff_diff = -1.90
            dof_pos_offset = 0.24
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
            feet_swing_clearance = -0.35
            feet_step_length_symmetry = -0.12
            feet_air_time_symmetry = -0.12
            double_support_stamping = -0.35

            ankle_action_abs = -0.08
            ankle_action_rate = -0.28
            ankle_pitch_pos_offset = -0.12
            ankle_pitch_motion = 0.035

    class normalization(GR2MainBodyCfg.normalization):
        actions_min = numpy.array([
            -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
            -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
        ])
        actions_max = numpy.array([
            2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
            2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
        ])

        clip_observations = 100.0

        clip_actions_min = actions_min - numpy.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ])
        clip_actions_max = actions_max + numpy.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ])

    class mirror(GR2MainBodyCfg.mirror):
        enable_mirror = False


class GR2DynamicWalk12CfgPPO(GR2BaseCfgPPO, GR2DynamicWalk12Cfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2BaseCfgPPO.runner):
        experiment_name = "GR2LowerBody12"
        num_steps_per_env = 64

        run_name = "dynamic_walk_12_vendor_contract"
        max_iterations = 5000
        save_interval = 100

        resume = False
        load_run = -1
        checkpoint = -1

    class algorithm(GR2BaseCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 25
        learning_rate = 1.0e-4
        learning_rate_min = 1.0e-5
        learning_rate_max = 1.0e-3
        schedule = "adaptive"
        desired_kl = 0.03

        storage_class = "RolloutStorage"

    class policy(GR2BaseCfgPPO.policy):
        class_name = "ActorCriticMLP"

        actor_hidden_dims = [1024, 512, 256, 128]
        critic_hidden_dims = [1024, 512, 256, 128]
        activation = "elu"
        init_weights = False

        fixed_std = False
        init_noise_std = [0.25] * GR2DynamicWalk12Cfg.env.num_actions

        decay_std = False
        decay_ratio = 1 - 2.0e-6
        decay_std_min = 0.08
