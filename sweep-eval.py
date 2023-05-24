from ray import tune

from evaluate import sweep

sweep(seed=tune.grid_search(list(range(4))))
