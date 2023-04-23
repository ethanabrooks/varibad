from ray import tune

from main import sweep

sweep(seed=tune.grid_search(list(range(8))))
