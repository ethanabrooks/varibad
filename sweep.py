import datetime
import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

import torch

import main
from utils.helpers import get_project_name


def task(seed, args, rest_args):
    try:
        print("Running task with seed", seed)
        timestamp = datetime.datetime.now().strftime("-%d-%m-%H:%M:%S")
        args = main.parse_args(rest_args)
        args.exp_group = f"{args.env_name}-{args.exp_label}-{timestamp}"
        args.project_name = get_project_name()
        args.seed = seed
        args.device = args.seed % torch.cuda.device_count()
        main.train(args)
    except Exception as e:
        raise Exception("".join(traceback.format_exception(None, e, e.__traceback__)))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4)
    args, rest_args = parser.parse_known_args()
    futures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for i in range(args.num_seeds):
            future = executor.submit(task, i, args, rest_args)
            futures.append(future)

    # Check for exceptions in the completed futures
    for future in futures:
        if future.exception() is not None:
            print(f"Error occurred: {future.exception()}")
