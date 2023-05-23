import argparse
import torch
import torch.nn as nn
import os
import pickle
from main import parse_args
from metalearner import MetaLearner
from utils.evaluation import evaluate
from utils.tb_logger import TBLogger
import wandb
from pathlib import Path

from vae import VaribadVAE


def load_pickle(loadpath: Path):
    with loadpath.open("rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loadpath")
    parser.add_argument("--num_episodes", type=int)
    parser.add_argument("--test", action="store_true")
    args, rest_args = parser.parse_known_args()
    loadpath = args.loadpath
    num_episodes = args.num_episodes
    test = args.test
    args = parse_args(rest_args)
    args.test = test

    if not args.debug:
        wandb.init(
            project="In-Context Model-Based Planning",
            name=f"evaluate-{args.env_name}-{args.exp_label}",
            sync_tensorboard=True,
            notes=args.notes,
        )

    for seed in args.seed:
        args.seed = seed
        metalearner = MetaLearner(args)
        logger = metalearner.logger

        for name, obj, attr in [
            ("encoder", metalearner.vae, "encoder"),
            ("policy", metalearner.policy, "actor_critic"),
        ]:
            name = f"{name}.pt"
            wandb.restore(name, run_path=loadpath, root=logger.full_output_folder)
            module = torch.load(os.path.join(logger.full_output_folder, name))
            assert isinstance(module, nn.Module)
            setattr(obj, attr, module)

        evaluate(
            args,
            metalearner.policy,
            ret_rms=None,
            seed=seed,
            tasks=None,
            logger=logger,
            encoder=metalearner.vae.encoder,
            num_episodes=num_episodes,
        )
        print("=================== DONE =====================")


if __name__ == "__main__":
    main()
