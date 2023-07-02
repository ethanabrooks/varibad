from argparse import ArgumentParser
from ray import tune

from main import sweep

parser = ArgumentParser()
parser.add_argument("--start-seed", type=int, default=0)
parser.add_argument("--num-seeds", type=int, default=8)
args, rest_args = parser.parse_known_args()
sweep(
    config=dict(
        seed=tune.grid_search([s + args.start_seed for s in range(args.num_seeds)])
    ),
    args=rest_args,
)
