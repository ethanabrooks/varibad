import argparse
import wandb
import os
import pickle
from pathlib import Path
import json

import torch
import torch.nn as nn
from wandb.sdk.wandb_run import Run

import utils.helpers as utl
from main import parse_args as base_parse_args
from metalearner import MetaLearner
from utils import evaluation


def load_pickle(loadpath: Path):
    with loadpath.open("rb") as f:
        return pickle.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadpath")
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--test-threshold", type=float)
    args, rest_args = parser.parse_known_args()
    loadpath = args.loadpath
    num_episodes = args.num_episodes
    test_threshold = args.test_threshold
    args = base_parse_args(rest_args)
    args.loadpath = loadpath
    args.num_episodes = num_episodes
    args.test_threshold = test_threshold
    return args


def main():
    return evaluate(parse_args())


def evaluate(args):
    if not args.debug:
        config = {}
        for k, v in vars(args).items():
            try:
                json.dumps(v)
            except TypeError:
                continue
            config[k] = v
        wandb.init(
            name=f"evaluate-{args.env_name}-{args.exp_label}",
            config=config,
            notes=args.notes,
            project=utl.get_project_name(),
            sync_tensorboard=True,
            tags=utl.get_tags(args.max_rollouts_per_task),
        )

    seeds = args.seed
    if not isinstance(seeds, list):
        seeds = [seeds]
    args.store_rollouts = True
    for seed in seeds:
        args.seed = seed
        metalearner = MetaLearner(args)
        logger = metalearner.logger

        for name, obj, attr in [
            ("encoder", metalearner.vae, "encoder"),
            ("policy", metalearner.policy, "actor_critic"),
        ]:
            name = f"{name}.pt"
            wandb.restore(name, run_path=args.loadpath, root=logger.full_output_folder)
            module = torch.load(os.path.join(logger.full_output_folder, name))
            assert isinstance(module, nn.Module)
            setattr(obj, attr, module)

        evaluation.evaluate(
            args,
            metalearner.policy,
            ret_rms=None,
            seed=seed,
            tasks=None,
            logger=logger,
            encoder=metalearner.vae.encoder,
            num_episodes=args.num_episodes,
            visualize=True,
            test_threshold=args.test_threshold,
        )
    wandb.finish()
    print("=================== DONE =====================")


def sweep(**config):
    return utl.sweep(args=parse_args(), config=config, train_func=evaluate)


if __name__ == "__main__":
    main()
