# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # override some parameters for testing
    env_cfg.env.episode_length_s = 600.0
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False

    env_cfg.noise.add_noise = False

    env_cfg.domain_rand.randomize_friction = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    fixed_commands = _get_fixed_commands(env)
    if fixed_commands is not None:
        _apply_fixed_commands(env, fixed_commands)
        env.compute_observations()
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            'logs',
            train_cfg.runner.experiment_name,
            'exported',
        )
        path = export_policy_as_jit(ppo_runner.algorithm.actor_critic, path)
        print(
            f"\033[93m"
            f"EXPORT_POLICY: "
            f"Exported policy as jit script to: "
            f"{path}"
            f"\033[0m"
        )

    # -------------------------------------------------------

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100 if PLOT_STATES else -1  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = _get_vector_env("PLAY_CAMERA_POS", env_cfg.viewer.pos)
    camera_vel = np.array([1., 1., 0.])
    camera_lookat = _get_vector_env("PLAY_CAMERA_LOOKAT", env_cfg.viewer.lookat)
    camera_direction = camera_lookat - camera_position
    env.set_camera(camera_position, camera_lookat)
    img_idx = 0
    frame_dir = os.environ.get(
        "PLAY_FRAMES_DIR",
        os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
    )
    frame_stride = max(1, int(os.environ.get("PLAY_FRAME_STRIDE", "2")))
    record_seconds = float(os.environ.get("PLAY_RECORD_SECONDS", "0"))
    max_steps = 10 * int(env.max_episode_length)
    if record_seconds > 0:
        max_steps = min(max_steps, int(record_seconds / env.dt))
    if RECORD_FRAMES:
        os.makedirs(frame_dir, exist_ok=True)

    for i in range(max_steps):
        if fixed_commands is not None:
            _apply_fixed_commands(env, fixed_commands)
            env.compute_observations()
            obs = env.get_observations()

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        if fixed_commands is not None:
            _apply_fixed_commands(env, fixed_commands)
            env.compute_observations()
            obs = env.get_observations()

        if RECORD_FRAMES:
            if i % frame_stride == 0:
                filename = os.path.join(frame_dir, f"{img_idx:06d}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            command_x = env.commands[robot_index, 0].item() if env.commands.shape[1] > 0 else 0.0
            command_y = env.commands[robot_index, 1].item() if env.commands.shape[1] > 1 else 0.0
            command_yaw = env.commands[robot_index, 2].item() if env.commands.shape[1] > 2 else 0.0
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': command_x,
                    'command_y': command_y,
                    'command_yaw': command_yaw,
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )

        elif PLOT_STATES and i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


def _get_fixed_commands(env):
    command_values = [
        os.environ.get("PLAY_COMMAND_X"),
        os.environ.get("PLAY_COMMAND_Y"),
        os.environ.get("PLAY_COMMAND_YAW"),
    ]
    if all(value is None for value in command_values):
        return None

    commands = torch.zeros(env.num_envs, env.cfg.commands.num_commands, device=env.device)
    for command_idx, value in enumerate(command_values):
        if value is not None and command_idx < commands.shape[1]:
            commands[:, command_idx] = float(value)

    print(
        "PLAY_COMMAND override: "
        f"x={commands[0, 0].item() if commands.shape[1] > 0 else 0.0:.3f}, "
        f"y={commands[0, 1].item() if commands.shape[1] > 1 else 0.0:.3f}, "
        f"yaw={commands[0, 2].item() if commands.shape[1] > 2 else 0.0:.3f}"
    )
    return commands


def _apply_fixed_commands(env, commands):
    env_ids = torch.arange(env.num_envs, device=env.device)
    if hasattr(env, "set_commands"):
        env.set_commands(env_ids, commands)
    else:
        env.commands[:] = commands


def _get_vector_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return np.array(default, dtype=np.float64)

    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must have three comma-separated values, got: {value}")
    vector = np.array(parts, dtype=np.float64)
    print(f"{name} override: {vector.tolist()}")
    return vector


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = os.environ.get("PLAY_RECORD_FRAMES", "0") == "1"
    PLOT_STATES = os.environ.get("PLAY_PLOT_STATES", "1") == "1"
    MOVE_CAMERA = False
    args = get_args()
    play(args)
