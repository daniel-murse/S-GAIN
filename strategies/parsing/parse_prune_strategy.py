
from strategies.pruning.magnitude_prune_strategy import MagnitudePruneStrategy
from strategies.pruning.random_prune_strategy import RandomPruneStrategy


def create_prune_strategy(mode_and_params, sparsity, w_counts):
    mode, params = mode_and_params

    if mode in ("none"):
        return None
    elif mode in ("prune_mask_by_random", "prune_mask_by_magnitude"):
        if len(params) != 4 or params[0] != "period" or params[2] != "percentage_active":
            print("Invalid initialisation strategy, missing period", mode_and_params)
            raise ValueError
        period = int(params[1])
        percentage_active = float(params[2])
        
        return MagnitudePruneStrategy(sparsity, )
    else:
        print("Invalid initialisation strategy", mode_and_params)
        raise ValueError