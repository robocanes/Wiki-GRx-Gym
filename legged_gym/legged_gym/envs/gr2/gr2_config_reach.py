import numpy

from legged_gym.envs.gr2.gr2_config import (
    GR2Cfg as GR2BaseCfg,
    GR2CfgPPO as GR2BaseCfgPPO,
)


class GR2ReachCfg(GR2BaseCfg):
    class env(GR2BaseCfg.env):
        episode_length_s = 6

        num_envs = 8192
        num_obs = 101
        num_pri_obs = 114
        num_actions = 29

        use_stack = True
        num_stack = 5

    class reach:
        # Target is sampled in the robot base frame. +x is forward, +y is left.
        target_x_range = [0.18, 0.38]
        target_y_abs_range = [0.10, 0.25]
        target_z_range = [-0.15, 0.375]
        target_roll_range = [-0.20, 0.20]
        target_pitch_range = [-0.20, 0.20]
        target_yaw_range = [-0.35, 0.35]
        draw_target_volume_box = True

    class asset(GR2BaseCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/GR2/urdf/GR2_raw.urdf"
        terminate_project_gravity_less_than = 0.82

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
        mesh_type = "plane"
        curriculum = False
        measure_heights = False

    class viewer(GR2BaseCfg.viewer):
        pos = [3.0, 0.0, 1.35]
        lookat = [0.0, 0.0, 0.95]

    class commands(GR2BaseCfg.commands):
        command_profile = "not_use"
        curriculum = False
        num_commands = 1
        gait_patterns = []
        resample_command_interval_s = 1.5

    class control(GR2BaseCfg.control):
        action_names = [
            # legs
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee_pitch",
            "ankle_pitch",
            "ankle_roll",

            # waist
            "waist_yaw",

            # head
            "head_yaw",
            "head_pitch",

            # arms
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow_pitch",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]

        action_scale = {
            "hip_pitch": 0.12,
            "hip_roll": 0.08,
            "hip_yaw": 0.08,
            "knee_pitch": 0.12,
            "ankle_pitch": 0.10,
            "ankle_roll": 0.08,

            "waist_yaw": 0.25,

            "head_yaw": 0.35,
            "head_pitch": 0.22,

            "shoulder_pitch": 0.35,
            "shoulder_roll": 0.22,
            "shoulder_yaw": 0.28,
            "elbow_pitch": 0.28,
            "wrist_yaw": 0.18,
            "wrist_pitch": 0.18,
            "wrist_roll": 0.18,
        }

        dof_pos_offset_scale = {
            "waist_yaw": 0.75,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.30,
            "shoulder_roll": 0.30,
            "shoulder_yaw": 0.30,
            "elbow_pitch": 0.30,
            "wrist_yaw": 0.50,
            "wrist_pitch": 0.50,
            "wrist_roll": 0.50,

            "hip_pitch": 1.5,
            "hip_roll": 1.5,
            "hip_yaw": 1.5,
            "knee_pitch": 1.5,
            "ankle_pitch": 1.5,
            "ankle_roll": 1.5,
        }

        dof_vel_scale = {
            "waist_yaw": 0.75,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.40,
            "shoulder_roll": 0.40,
            "shoulder_yaw": 0.40,
            "elbow_pitch": 0.40,
            "wrist_yaw": 0.60,
            "wrist_pitch": 0.60,
            "wrist_roll": 0.60,

            "hip_pitch": 1.5,
            "hip_roll": 1.5,
            "hip_yaw": 1.5,
            "knee_pitch": 1.5,
            "ankle_pitch": 1.5,
            "ankle_roll": 1.5,
        }

        dof_tor_scale = {
            "waist_yaw": 0.75,

            "head_yaw": 1.0,
            "head_pitch": 1.0,

            "shoulder_pitch": 0.40,
            "shoulder_roll": 0.40,
            "shoulder_yaw": 0.40,
            "elbow_pitch": 0.40,
            "wrist_yaw": 0.60,
            "wrist_pitch": 0.60,
            "wrist_roll": 0.60,

            "hip_pitch": 1.5,
            "hip_roll": 1.5,
            "hip_yaw": 1.5,
            "knee_pitch": 1.5,
            "ankle_pitch": 1.5,
            "ankle_roll": 1.5,
        }

    class rewards(GR2BaseCfg.rewards):
        reach_pos_tracking_sigma = 0.08
        reach_orient_tracking_sigma = 0.75
        reach_success_distance = 0.055
        reach_upright_gate = 0.90
        reach_stable_lin_vel_sigma = 0.18
        reach_stable_ang_vel_sigma = 0.45
        reach_ee_vel_sigma = 0.35

        only_positive_rewards = False

        class scales(GR2BaseCfg.rewards.scales):
            reach_target_pos = 6.00
            reach_target_orient = 0.10
            reach_target_success = 1.00
            reach_target_stable = 2.50
            active_end_effector_still = -0.65
            head_look_at_target = 0.55

            base_still = -3.00
            main_body_dof_pos = -1.20
            inactive_arm_still = -0.12

            base_height_offset_range = 1.25
            base_flat_orient = 1.80
            torso_flat_orient = 1.80

            action_diff = -0.45
            action_diff_diff = -0.10
            dof_acc = -0.03
            dof_tor = -0.03
            termination = -8.00

            limits_dof_pos_without_ankle = -4.00
            limits_dof_vel_without_ankle = -1.00
            limits_dof_tor = -0.25

    class noise(GR2BaseCfg.noise):
        add_noise = True

        class noise_scales(GR2BaseCfg.noise.noise_scales):
            target_pos = 0.01
            target_rpy = 0.01

    class normalization(GR2BaseCfg.normalization):
        actions_min = numpy.array([
            -2.6180, -0.5934, -0.6981, -0.0873, -0.7854, -0.38397,
            -2.6180, -1.5708, -1.5708, -0.0873, -0.7854, -0.38397,
            -2.6180,
            -1.3963, -0.5236,
            -2.9671, -0.5236, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
            -2.9671, -2.7925, -1.8326, -1.5272, -1.8326, -0.6109, -0.9600,
        ])
        actions_max = numpy.array([
            2.6180, 1.5708, 1.5708, 2.3562, 0.7854, 0.38397,
            2.6180, 0.5934, 0.6981, 2.3562, 0.7854, 0.38397,
            2.6180,
            1.3963, 0.5236,
            2.9671, 2.7925, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
            2.9671, 0.5236, 1.8326, 0.4800, 1.8326, 0.6109, 0.9600,
        ])

        clip_observations = 100.0
        clip_actions_min = actions_min - numpy.ones_like(actions_min)
        clip_actions_max = actions_max + numpy.ones_like(actions_max)

    class mirror(GR2BaseCfg.mirror):
        enable_mirror = False
        observations_coefficient = numpy.ones(101)
        observations_exchange = numpy.array([])
        actions_coefficient = numpy.ones(29)
        actions_exchange = numpy.array([])


class GR2ReachCfgPPO(GR2BaseCfgPPO, GR2ReachCfg):
    runner_class_name = "OnPolicyRunner"

    class runner(GR2BaseCfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "front_random_wrist_target_headlook_nojitter_highlowz"
        max_iterations = 5000
        save_interval = 100

        resume = False
        load_run = -1
        checkpoint = -1

    class algorithm(GR2BaseCfgPPO.algorithm):
        class_name = "PPO"

        num_learning_epochs = 8
        num_mini_batches = 32
        learning_rate = 5.e-5
        learning_rate_min = 1.e-5
        learning_rate_max = 5.e-4
        schedule = "adaptive"
        desired_kl = 0.03

        storage_class = "RolloutStorage"

    class policy(GR2BaseCfgPPO.policy):
        init_noise_std = [
            0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
            0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
            0.20,
            0.18, 0.16,
            0.35, 0.30, 0.30, 0.30, 0.25, 0.25, 0.25,
            0.35, 0.30, 0.30, 0.30, 0.25, 0.25, 0.25,
        ]
