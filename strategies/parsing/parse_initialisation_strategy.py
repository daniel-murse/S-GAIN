

from strategies.initialisation.magnitude_initialisation_strategy import MagnitudeInitialisationStrategy
from strategies.initialisation.random_initialisation_strategy import RandomInitialisationStrategy


def create_initialisation_strategy(mode_and_params, sparsity):
    mode, params = mode_and_params

    if mode in ("none"):
        return None
    elif mode in ("init_mask_by_prune_random"):
        return RandomInitialisationStrategy(sparsity)
    elif mode in ("init_mask_by_prune_magnitude"):
        return MagnitudeInitialisationStrategy(sparsity)
    else:
        print("Invalid initialisation strategy", mode_and_params)
        raise ValueError