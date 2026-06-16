from legged_gym.envs.gr2.gr2_config_reach_planted_curriculum import (
    GR2ReachPlantedCurriculumZ1Cfg,
    GR2ReachPlantedCurriculumZ1CfgPPO,
)


class GR2ReachPlantedCurriculumZ2Cfg(GR2ReachPlantedCurriculumZ1Cfg):
    """Boundary-biased reach stage, initialized from the stable planted Z1 policy."""

    class reach(GR2ReachPlantedCurriculumZ1Cfg.reach):
        # Keep forward reach at the tested limit, but push higher/lower and slightly wider.
        target_x_range = [0.16, 0.42]
        target_y_abs_range = [0.10, 0.35]
        target_z_range = [-0.26, 0.65]

        # Bias training toward the newly expanded boundaries while preserving interior samples.
        target_edge_sample_prob = 0.72
        target_edge_z_band_fraction = 0.22
        target_edge_y_band_fraction = 0.28
        target_edge_x_band_fraction = 0.25
        target_edge_low_z_weight = 0.32
        target_edge_high_z_weight = 0.36
        target_edge_outer_y_weight = 0.22
        target_edge_forward_x_weight = 0.10

    class rewards(GR2ReachPlantedCurriculumZ1Cfg.rewards):
        pickup_low_target_start_z = 0.04
        pickup_low_target_full_z = -0.24
        pickup_base_xy_deadband = 0.050
        pickup_feet_xy_deadband = 0.040
        pickup_knee_flexion_target = 0.38
        pickup_hip_flexion_target = 0.19

        class scales(GR2ReachPlantedCurriculumZ1Cfg.rewards.scales):
            reach_target_pos = 7.35
            reach_target_success = 1.25
            reach_target_stable = 3.35
            head_look_at_target = 0.90

            base_still = -3.00
            base_xy_displacement = -1.35
            feet_xy_displacement = -1.00
            feet_speed_xy = -0.26
            low_target_knee_flexion = 6.25
            low_target_hip_flexion = 2.35

            base_height_offset_range = 0.38
            base_flat_orient = 1.80
            torso_flat_orient = 2.00


class GR2ReachPlantedCurriculumZ2CfgPPO(
    GR2ReachPlantedCurriculumZ1CfgPPO,
    GR2ReachPlantedCurriculumZ2Cfg,
):
    class runner(GR2ReachPlantedCurriculumZ1CfgPPO.runner):
        experiment_name = "GR2Reach"
        num_steps_per_env = 48

        run_name = "planted_curriculum_z2_edges_from_z1_11400"
        max_iterations = 700
        save_interval = 100

        resume = True
        load_run = "Jun16_17-36-56_planted_curriculum_z1_from_expanded_stable"
        checkpoint = 11400

    class algorithm(GR2ReachPlantedCurriculumZ1CfgPPO.algorithm):
        learning_rate = 1.2e-5
        learning_rate_min = 4.e-6
        learning_rate_max = 8.0e-5
        desired_kl = 0.014

    class policy(GR2ReachPlantedCurriculumZ1CfgPPO.policy):
        init_noise_std = [0.07] * GR2ReachPlantedCurriculumZ2Cfg.env.num_actions
