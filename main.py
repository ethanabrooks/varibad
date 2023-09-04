"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
from torchsnapshot import Snapshot
import argparse
import json
import os
import warnings
from typing import Optional

import numpy as np
import torch
from git import Repo
from utils.tb_logger import TBLogger
from wandb.sdk.wandb_run import Run

from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import LazyMemmapStorage
import wandb
from config import config

# get configs
from environments.parallel_envs import make_vec_envs
from learner import Learner, get_num_updates
from metalearner import MetaLearner
from utils import helpers as utl
from utils.helpers import get_project_name, get_tags


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", default="gridworld_varibad")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--replay-buffer", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--artifact")
    parser.add_argument("--gpus-per-proc", type=float, default=1)
    args = parser.parse_args(args)
    env = args.env_type

    # --- GridWorld ---

    # if env == "gridworld_belief_oracle":
    #     args = args_grid_belief_oracle.get_args(rest_args)
    # elif env == "gridworld_varibad":
    #     args = args_grid_varibad.get_args(rest_args)
    # elif env == "gridworld_rl2":
    #     args = args_grid_rl2.get_args(rest_args)

    # --- Alchemy ---

    if env == "alchemy_ad":
        config_args = config.AlchemyAD
    elif env == "alchemy_multitask":
        config_args = config.AlchemyMultitask
    elif env == "alchemy_rl2":
        config_args = config.AlchemyRL2

    # --- MUJOCO ---
    #
    # - AntDir -
    elif env == "ant_dir_ad":
        config_args = config.AntDirAD
    elif env == "ant_dir_expert":
        config_args = config.AntDirExpert
    elif env == "ant_dir_multitask":
        config_args = config.AntDirMultitask
    # elif env == "ant_dir_varibad":
    #     args = args_ant_dir_varibad.get_args(rest_args)
    elif env == "ant_dir_rl2":
        config_args = config.AntDirRL2
    #
    # - AntGoal -
    elif env == "ant_goal_ad":
        config_args = config.AntGoalAD
    elif env == "ant_goal_expert":
        config_args = config.AntGoalExpert
    elif env == "ant_goal_multitask":
        config_args = config.AntGoalMultitask
    # elif env == "ant_goal_varibad":
    #     args = args_ant_goal_varibad.get_args(rest_args)
    # elif env == "ant_goal_humplik":
    #     args = args_ant_goal_humplik.get_args(rest_args)
    elif env == "ant_goal_rl2":
        config_args = config.AntGoalRL2

    # - CheetahDir -
    elif env == "cheetah_dir_ad":
        config_args = config.CheetahDirAD
    elif env == "cheetah_dir_multitask":
        config_args = config.CheetahDirMultitask
    elif env == "cheetah_dir_expert":
        config_args = config.CheetahDirExpert
    # elif env == "cheetah_dir_varibad":
    #     args = args_cheetah_dir_varibad.get_args(rest_args)
    elif env == "cheetah_dir_rl2":
        config_args = config.CheetahDirRL2
    #
    # - CheetahVel -
    elif env == "cheetah_vel_ad":
        config_args = config.CheetahVelAD
    elif env == "cheetah_vel_multitask":
        config_args = config.CheetahVelMultitask
    elif env == "cheetah_vel_expert":
        config_args = config.CheetahVelExpert
    # elif env == "cheetah_vel_avg":
    #     args = args_cheetah_vel_avg.get_args(rest_args)
    # elif env == "cheetah_vel_varibad":
    #     args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == "cheetah_vel_rl2":
        config_args = config.CheetahVelRL2
    #
    # - Walker -
    elif env == "walker_ad":
        config_args = config.WalkerAD
    elif env == "walker_multitask":
        config_args = config.WalkerMultitask
    elif env == "walker_expert":
        config_args = config.WalkerExpert
    # elif env == "walker_avg":
    #     args = args_walker_avg.get_args(rest_args)
    # elif env == "walker_varibad":
    #     args = args_walker_varibad.get_args(rest_args)
    elif env == "walker_rl2":
        config_args = config.WalkerRL2
    #
    # - Hopper -
    elif env == "hopper_multitask":
        config_args = config.HopperMultitask
    elif env == "hopper_expert":
        config_args = config.HopperExpert
    # elif env == "hopper_avg":
    #     args = args_walker_avg.get_args(rest_args)
    # elif env == "hopper_varibad":
    #     args = args_walker_varibad.get_args(rest_args)
    elif env == "hopper_rl2":
        config_args = config.HopperRL2
    #
    # - HumanoidDir -
    # elif env == "humanoid_dir_multitask":
    #     args = args_humanoid_dir_multitask.get_args(rest_args)
    # elif env == "humanoid_dir_expert":
    #     args = args_humanoid_dir_expert.get_args(rest_args)
    # elif env == "humanoid_dir_varibad":
    #     args = args_humanoid_dir_varibad.get_args(rest_args)
    # elif env == "humanoid_dir_rl2":
    #     args = args_humanoid_dir_rl2.get_args(rest_args)
    # else:
    #     raise Exception("Invalid Environment")

    # --- PointRobot 2D Navigation ---
    elif env == "point_robot_multitask":
        config_args = config.PointRobotMultitask
    elif env == "point_robot_rl2":
        config_args = config.PointRobotRL2
    elif env == "point_robot_ad":
        config_args = config.PointRobotAD

    elif env == "sparse_point_robot_multitask":
        config_args = config.SparsePointRobotMultitask
    elif env == "sparse_point_robot_rl2":
        config_args = config.SparsePointRobotRL2
    elif env == "sparse_point_robot_ad":
        config_args = config.SparsePointRobotAD

    elif env == "grid_rl2":
        config_args = config.GridRL2

    else:
        raise Exception(f"Invalid Environment: {env}")

    args.commit = Repo(".").head.commit.hexsha
    config_args = config_args()
    for k, v in vars(args).items():
        setattr(config_args, k, v)

    return config_args


def main():
    args = parse_args()
    args.project_name = get_project_name()
    return train(args)


def train(args, run: Optional[Run] = None):
    # warning for deterministic execution
    if args.deterministic_execution:
        print("Envoking deterministic code execution.")
        if torch.backends.cudnn.enabled:
            warnings.warn("Running with deterministic CUDNN.")
        if args.num_processes > 1:
            raise RuntimeError(
                "If you want fully deterministic code, run it with num_processes=1."
                "Warning: This will slow things down and might break A2C if "
                "policy_num_steps < env._max_episode_steps."
            )

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(
            env_name=args.env_name,
            seed=0,
            num_processes=args.num_processes,
            gamma=args.policy_gamma,
            device="cpu",
            episodes_per_task=args.max_rollouts_per_task,
            normalise_rew=args.norm_rew_for_policy,
            ret_rms=None,
            tasks=None,
        )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, "decode_only_past") and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed

    if not args.debug:
        config = {}
        for k, v in vars(args).items():
            try:
                json.dumps(v)
            except TypeError:
                continue
            config[k] = v

        wandb.init(
            project=args.project_name,
            config=config,
            name=f"{args.env_name}-{args.exp_label}",
            sync_tensorboard=True,
            tags=get_tags(args.max_rollouts_per_task),
            notes=args.notes,
        )
        run = wandb.run

    # initialise tensorboard logger
    logger = TBLogger(args, args.exp_label, debug=args.debug)

    # initialise replay buffer
    if args.replay_buffer:
        replay_buffer_size = (
            len(seed_list)
            * get_num_updates(
                args.num_frames, args.policy_num_steps, args.num_processes
            )
            * args.policy_num_steps
        )
        replay_buffers = []

        for i in range(1 if args.single_task_mode else args.num_processes):
            replay_buffer_path = os.path.join(
                logger.full_output_folder, "replay-buffers", str(i)
            )
            if not os.path.exists(replay_buffer_path):
                os.makedirs(replay_buffer_path)
            replay_buffers.append(
                ReplayBuffer(
                    LazyMemmapStorage(
                        replay_buffer_size, scratch_dir=replay_buffer_path
                    )
                )
            )

    else:
        replay_buffers = None

    for seed in seed_list:
        print("training", seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(
                args, logger=logger, replay_buffers=replay_buffers, run=run
            )
        else:
            learner = MetaLearner(args)
        learner.train()

        if args.replay_buffer:
            print("Replay buffer sizes:")
            for i, buffer in enumerate(replay_buffers):
                print(f"{i}: {len(buffer)} / {replay_buffer_size}")
            print()

    save_path = os.path.join(logger.full_output_folder, "models")
    if args.replay_buffer:
        replay_buffer_save_path = os.path.join(save_path, "replay-buffers")
        Snapshot.take(
            path=replay_buffer_save_path,
            app_state={str(i): b for i, b in enumerate(replay_buffers)},
        )
        print(f"Saved replay-buffer to: {replay_buffer_save_path}")

        if run is not None:
            artifact = wandb.Artifact(
                name=f"{args.env_name}-{args.exp_label}",
                type="dataset",
            )
            artifact.add_dir(replay_buffer_save_path)
            run.log_artifact(artifact, aliases=[f"{args.artifact}.{args.seed}"])
    wandb.finish()


def sweep(config: dict, args: list[str]):
    return utl.sweep(args=parse_args(args), config=config, train_func=train)


if __name__ == "__main__":
    main()
