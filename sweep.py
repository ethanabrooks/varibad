from argparse import ArgumentParser

from ray import tune

from main import sweep

parser = ArgumentParser()
parser.add_argument("--seq-len", type=int, default=16)
parser.add_argument("--num-procs", type=int, default=16)
args, rest_args = parser.parse_known_args()
sweep(
    config=dict(
        seed=tune.grid_search(
            [
                list(range(i, i + args.seq_len))
                for i in range(0, args.seq_len * args.num_procs, args.seq_len)
            ]
        )
    ),
    args=rest_args,
)
