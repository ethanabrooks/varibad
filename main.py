"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import datetime
import time
import urllib
import warnings
from git import Repo
from typing import Optional

import numpy as np
import tomli
import torch
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from wandb.sdk.wandb_run import Run

import wandb
from config.alchemy import args_alchemy_multitask, args_alchemy_rl2

# get configs
from config.gridworld import args_grid_belief_oracle, args_grid_rl2, args_grid_varibad
from config.mujoco import (
    args_ant_dir_expert,
    args_ant_dir_multitask,
    args_ant_dir_rl2,
    args_ant_dir_varibad,
    args_ant_goal_expert,
    args_ant_goal_humplik,
    args_ant_goal_multitask,
    args_ant_goal_rl2,
    args_ant_goal_varibad,
    args_cheetah_dir_expert,
    args_cheetah_dir_multitask,
    args_cheetah_dir_rl2,
    args_cheetah_dir_varibad,
    args_cheetah_vel_avg,
    args_cheetah_vel_expert,
    args_cheetah_vel_multitask,
    args_cheetah_vel_rl2,
    args_cheetah_vel_varibad,
    args_humanoid_dir_expert,
    args_humanoid_dir_multitask,
    args_humanoid_dir_rl2,
    args_humanoid_dir_varibad,
    args_walker_avg,
    args_walker_expert,
    args_walker_multitask,
    args_walker_rl2,
    args_walker_varibad,
)
from config.pointrobot import (
    args_pointrobot_humplik,
    args_pointrobot_multitask,
    args_pointrobot_rl2,
    args_pointrobot_varibad,
)
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner

def get_project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


def get_tags(max_rollouts_per_task: Optional[int]):
    tags = ["multi-replay-buffers"]
    if max_rollouts_per_task is None:
        tags += ["single-task-histories"]
    return tags


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-type", default="gridworld_varibad")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--replay_buffer", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--artifact")
    args, rest_args = parser.parse_known_args(args)
    use_replay_buffer = args.replay_buffer
    debug = args.debug
    env = args.env_type
    notes = args.notes
    artifact = args.artifact

    # --- GridWorld ---

    if env == "gridworld_belief_oracle":
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == "gridworld_varibad":
        args = args_grid_varibad.get_args(rest_args)
    elif env == "gridworld_rl2":
        args = args_grid_rl2.get_args(rest_args)

    # --- Alchemy ---

    if env == "alchemy_multitask":
        args = args_alchemy_multitask.get_args(rest_args)
    elif env == "alchemy_rl2":
        args = args_alchemy_rl2.get_args(rest_args)

    # --- PointRobot 2D Navigation ---

    elif env == "pointrobot_multitask":
        args = args_pointrobot_multitask.get_args(rest_args)
    elif env == "pointrobot_varibad":
        args = args_pointrobot_varibad.get_args(rest_args)
    elif env == "pointrobot_rl2":
        args = args_pointrobot_rl2.get_args(rest_args)
    elif env == "pointrobot_humplik":
        args = args_pointrobot_humplik.get_args(rest_args)

    # --- MUJOCO ---

    # - CheetahDir -
    elif env == "cheetah_dir_multitask":
        args = args_cheetah_dir_multitask.get_args(rest_args)
    elif env == "cheetah_dir_expert":
        args = args_cheetah_dir_expert.get_args(rest_args)
    elif env == "cheetah_dir_varibad":
        args = args_cheetah_dir_varibad.get_args(rest_args)
    elif env == "cheetah_dir_rl2":
        args = args_cheetah_dir_rl2.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == "cheetah_vel_multitask":
        args = args_cheetah_vel_multitask.get_args(rest_args)
    elif env == "cheetah_vel_expert":
        args = args_cheetah_vel_expert.get_args(rest_args)
    elif env == "cheetah_vel_avg":
        args = args_cheetah_vel_avg.get_args(rest_args)
    elif env == "cheetah_vel_varibad":
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == "cheetah_vel_rl2":
        args = args_cheetah_vel_rl2.get_args(rest_args)
    #
    # - AntDir -
    elif env == "ant_dir_multitask":
        args = args_ant_dir_multitask.get_args(rest_args)
    elif env == "ant_dir_expert":
        args = args_ant_dir_expert.get_args(rest_args)
    elif env == "ant_dir_varibad":
        args = args_ant_dir_varibad.get_args(rest_args)
    elif env == "ant_dir_rl2":
        args = args_ant_dir_rl2.get_args(rest_args)
    #
    # - AntGoal -
    elif env == "ant_goal_multitask":
        args = args_ant_goal_multitask.get_args(rest_args)
    elif env == "ant_goal_expert":
        args = args_ant_goal_expert.get_args(rest_args)
    elif env == "ant_goal_varibad":
        args = args_ant_goal_varibad.get_args(rest_args)
    elif env == "ant_goal_humplik":
        args = args_ant_goal_humplik.get_args(rest_args)
    elif env == "ant_goal_rl2":
        args = args_ant_goal_rl2.get_args(rest_args)
    #
    # - Walker -
    elif env == "walker_multitask":
        args = args_walker_multitask.get_args(rest_args)
    elif env == "walker_expert":
        args = args_walker_expert.get_args(rest_args)
    elif env == "walker_avg":
        args = args_walker_avg.get_args(rest_args)
    elif env == "walker_varibad":
        args = args_walker_varibad.get_args(rest_args)
    elif env == "walker_rl2":
        args = args_walker_rl2.get_args(rest_args)
    #
    # - HumanoidDir -
    elif env == "humanoid_dir_multitask":
        args = args_humanoid_dir_multitask.get_args(rest_args)
    elif env == "humanoid_dir_expert":
        args = args_humanoid_dir_expert.get_args(rest_args)
    elif env == "humanoid_dir_varibad":
        args = args_humanoid_dir_varibad.get_args(rest_args)
    elif env == "humanoid_dir_rl2":
        args = args_humanoid_dir_rl2.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    args.use_replay_buffer = use_replay_buffer
    args.debug = debug
    args.notes = notes
    args.artifact = artifact
    args.commit = Repo(".").head.commit.hexsha
    return args


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
        wandb.init(
            project=args.project_name,
            name=f"{args.env_name}-{args.exp_label}",
            sync_tensorboard=True,
            tags=get_tags(args.max_rollouts_per_task),
            notes=args.notes,
        )
        run = wandb.run

    for seed in seed_list:
        print("training", seed)
        args.seed = seed
        args.action_space = None

        if args.disable_metalearner:
            # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
            # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
            learner = Learner(
                args,
                use_replay_buffer=args.use_replay_buffer,
                debug=args.debug,
                run=run,
            )
        else:
            learner = MetaLearner(args)
        learner.train()
    wandb.finish()


def sweep(**config):
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("-%d-%m-%H:%M:%S")
    group = f"{args.env_name}-{timestamp}"
    args.project_name = get_project_name()

    def train_func(sweep_params):
        for k, v in sweep_params.items():
            setattr(args, k, v)
        sleep_time = 1
        while True:
            try:
                run = setup_wandb(
                    config=vars(args),
                    group=group,
                    project=args.project_name,
                    rank_zero_only=False,
                    tags=get_tags(args.max_rollouts_per_task),
                    notes=args.notes,
                )
                break
            except wandb.errors.CommError:
                time.sleep(sleep_time)
                sleep_time *= 2
        print(
            f"wandb: ️👪 View group at {run.get_project_url()}/groups/{urllib.parse.quote(group)}/workspace"
        )
        return train(args, run=run)

    tune.Tuner(
        trainable=tune.with_resources(train_func, dict(gpu=1)),
        param_space=config,
    ).fit()


if __name__ == "__main__":
    main()
