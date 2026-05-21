import numpy

from legged_gym.envs.gr2.gr2_config import (
    GR2Cfg as GR2BaseCfg,
    GR2CfgPPO as GR2BaseCfgPPO,
)


class GR2UpperBodyCfg(GR2BaseCfg):
    class env(GR2BaseCfg.env):
        episode_length_s = 20

        # GR2_raw has 29 movable DOFs. This task controls legs, waist, shoulders,
        # and elbows, while head and wrist joints are held by their default PD targets.
        num_obs = 88
        num_pri_obs = 221
        num_actions = (1 + 4 + 4 + 6 + 6)

        use_stack = True
        num_stack = 5

    class asset(GR2BaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/GR2/urdf/GR2_raw.urdf"

        terminate_contacts_on = [
            GR2BaseCfg.asset.imu_name,
            GR2BaseCfg.asset.torso_name,
            GR2BaseCfg.asset.waist_name,
            GR2BaseCfg.asset.thigh_name,
            GR2BaseCfg.asset.hand_name,
            GR2BaseCfg.asset.end_effector_name,
            GR2BaseCfg.asset.payload_name,
        ]

    class terrain(GR2BaseCfg.terrain):
        mesh_type = "trimesh"
        curriculum = False
        num_rows = 2
        num_cols = 4
        max_init_terrain_level = 0

        # Terrain types:
        # smooth plane, rough plane,
        # smooth slope, rough slope,
        # smooth wave, rough wave,
        # smooth stairs, rough stairs,
        # stones, discrete,
        # gap, pit.
        # Keep walking mostly on flat ground with a little rough-plane variation.
        terrain_proportions = [
            0.85, 0.15,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
            0.0, 0.0,
        ]

    class control(GR2BaseCfg.control):
        action_names = [
            "waist_yaw",

            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",

            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_pitch",
            "ankle_roll",
        ]

        action_scale = {
            "waist_yaw": 0.75,

            "shoulder_pitch": 0.20,
            "shoulder_roll": 0.06,
            "shoulder_yaw": 0.06,
            "elbow_pitch": 0.10,

            "hip_pitch": 1.0,
            "hip_roll": 1.0,
            "hip_yaw": 1.0,
            "knee_pitch": 1.0,
            "ankle_pitch": 1.0,
            "ankle_roll": 1.0,
        }

        dof_pos_offset_scale = {
            "waist_yaw": 1.0,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.55,
            "shoulder_roll": 1.50,
            "shoulder_yaw": 1.50,
            "elbow_pitch": 0.45,
            "wrist_yaw": 1.0,
            "wrist_pitch": 1.0,
            "wrist_roll": 1.0,

            "hip_pitch": 1.0,
            "hip_roll": 1.0,
            "hip_yaw": 1.0,
            "knee_pitch": 1.0,
            "ankle_pitch": 1.0,
            "ankle_roll": 1.0,
        }

        dof_vel_scale = {
            "waist_yaw": 1.0,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.60,
            "shoulder_roll": 1.00,
            "shoulder_yaw": 1.00,
            "elbow_pitch": 0.55,
            "wrist_yaw": 1.0,
            "wrist_pitch": 1.0,
            "wrist_roll": 1.0,

            "hip_pitch": 1.0,
            "hip_roll": 1.0,
            "hip_yaw": 1.0,
            "knee_pitch": 1.0,
            "ankle_pitch": 1.0,
            "ankle_roll": 1.0,
        }

        dof_tor_scale = {
            "waist_yaw": 1.0,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.60,
            "shoulder_roll": 1.00,
            "shoulder_yaw": 1.00,
            "elbow_pitch": 0.55,
            "wrist_yaw": 1.0,
            "wrist_pitch": 1.0,
            "wrist_roll": 1.0,

            "hip_pitch": 1.0,
            "hip_roll": 1.0,
            "hip_yaw": 1.0,
            "knee_pitch": 1.0,
            "ankle_pitch": 1.0,
            "ankle_roll": 1.0,
        }

    class rewards(GR2BaseCfg.rewards):
        feet_air_time_target = 0.35

        class scales(GR2BaseCfg.rewards.scales):
            stand_still_dof_pos_waist_joint = 0.50
            stand_still_foot_distance = 0.25

            cmd_diff_base_lin_vel_x = 2.00
            cmd_diff_base_lin_vel_y = 0.50
            cmd_diff_base_ang_vel_yaw = 0.75

            base_lin_vel_z = 0.25
            base_height_offset_range = 0.50
            base_flat_orient = 0.25
            torso_flat_orient = 0.30

            action_diff = -7.00
            action_diff_diff = -1.50

            dof_pos_offset = 0.35
            shoulder_pitch_pos = -0.25
            shoulder_pitch_vel = -0.03
            shoulder_sideways_pos = -1.25
            shoulder_sideways_vel = -0.08
            dof_acc = -0.30
            dof_tor = -0.06

            limits_dof_pos_without_ankle = -10.00
            limits_dof_vel_without_ankle = -5.00
            limits_dof_tor = -1.00

            feet_speed_xy_close_to_ground = 0.20
            feet_force_z_close_to_ground = -0.20
            feet_stumble = -0.20
            feet_distance_too_close = -0.50
            feet_air_time = 2.00

    class normalization(GR2BaseCfg.normalization):
        # Order follows GR2_raw movable DOF order filtered by control.action_names:
        # left leg, right leg, waist, left arm shoulder/elbow, right arm shoulder/elbow.
        actions_min = numpy.array([
            -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
            -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
            -2.6180,
            -2.9671, -0.5236, -1.8326, -1.5272,
            -2.9671, -2.7925, -1.8326, -1.5272,
        ])
        actions_max = numpy.array([
            2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
            2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
            2.6180,
            2.9671, 2.7925, 1.8326, 0.4800,
            2.9671, 0.5236, 1.8326, 0.4800,
        ])

        clip_observations = 100.0
        clip_actions_min = actions_min - numpy.ones(21)
        clip_actions_max = actions_max + numpy.ones(21)

    class mirror(GR2BaseCfg.mirror):
        enable_mirror = False
        observations_coefficient = numpy.ones(88)
        observations_exchange = numpy.array([])
        actions_coefficient = numpy.ones(21)
        actions_exchange = numpy.array([])


class GR2UpperBodyCfgPPO(GR2BaseCfgPPO, GR2UpperBodyCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2BaseCfgPPO.runner):
        experiment_name = "GR2UpperBody"
        num_steps_per_env = 64

        run_name = "humanlike_flat_arms_quiet_elbows"
        max_iterations = 5000
        save_interval = 100

        resume = True
        load_run = "May20_21-30-42_humanlike_flat_arms_tighter_swing"
        checkpoint = 5800

    class algorithm(GR2BaseCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 25
        learning_rate = 5.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 5.e-4
        schedule = "adaptive"
        desired_kl = 0.03

        storage_class = "RolloutStorage"

    class policy(GR2BaseCfgPPO.policy):
        init_noise_std = [0.35] * GR2UpperBodyCfg.env.num_actions
