from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum_z2 import (
    GR2ReachPlantedCurriculumZ2Cfg,
    GR2ReachPlantedCurriculumZ2CfgPPO,
)


class GR2ReachPlantedCurriculumZ3Cfg(GR2ReachPlantedCurriculumZ2Cfg):
    """Ground-pick and overhead reach stage, initialized from the Z2 edge policy."""

    class reach(GR2ReachPlantedCurriculumZ2Cfg.reach):
        # Keep forward reach at the tested limit; extend vertical range toward floor/high shelf.
        target_x_range = [0.16, 0.42]
        target_y_abs_range = [0.10, 0.35]
        target_z_range = [-0.36, 0.78]

        target_edge_sample_prob = 0.80
        target_edge_z_band_fraction = 0.24
        target_edge_y_band_fraction = 0.25
        target_edge_x_band_fraction = 0.22
        target_edge_low_z_weight = 0.44
        target_edge_high_z_weight = 0.38
        target_edge_outer_y_weight = 0.12
        target_edge_forward_x_weight = 0.06

    class rewards(GR2ReachPlantedCurriculumZ2Cfg.rewards):
        pickup_low_target_start_z = 0.06
        pickup_low_target_full_z = -0.34
        pickup_base_xy_deadband = 0.048
        pickup_feet_xy_deadband = 0.038
        pickup_knee_flexion_target = 0.52
        pickup_hip_flexion_target = 0.26

        class scales(GR2ReachPlantedCurriculumZ2Cfg.rewards.scales):
            reach_target_pos = 7.55
            reach_target_success = 1.30
            reach_target_stable = 3.45
            head_look_at_target = 0.95

            base_still = -3.10
            base_xy_displacement = -1.55
            feet_xy_displacement = -1.15
            feet_speed_xy = -0.30
            low_target_knee_flexion = 7.20
            low_target_hip_flexion = 2.85

            base_height_offset_range = 0.42
            base_flat_orient = 1.85
            torso_flat_orient = 2.05

            action_diff = -0.46
            action_diff_diff = -0.105


class GR2ReachPlantedCurriculumZ3CfgPPO(
    GR2ReachPlantedCurriculumZ2CfgPPO,
    GR2ReachPlantedCurriculumZ3Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ2CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z3_ground_high_from_z2_12099"
        max_iterations = 600
        save_interval = 100

        resume = True
        load_run = "Jun16_18-36-30_planted_curriculum_z2_edges_from_z1_11400"
        checkpoint = 12099

    class algorithm(GR2ReachPlantedCurriculumZ2CfgPPO.algorithm):
        learning_rate = 1.0e-5
        learning_rate_min = 3.e-6
        learning_rate_max = 7.0e-5
        desired_kl = 0.013

    class policy(GR2ReachPlantedCurriculumZ2CfgPPO.policy):
        init_noise_std = [0.065] * GR2ReachPlantedCurriculumZ3Cfg.env.num_actions
