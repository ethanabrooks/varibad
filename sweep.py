from ray import tune

from main import sweep

sweep(
    num_stones_per_trial=tune.grid_search([1, 2, 3]),
    num_potions_per_trial=tune.grid_search([2, 6, 12]),
    max_steps_per_trial=tune.grid_search([20, 40]),
    end_trial_action=tune.grid_search([False, True]),
)
