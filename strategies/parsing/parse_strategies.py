import numpy as np
from strategies.initialisation.grasp_initialisation_strategy import GraSPInitialisationStrategy
from strategies.initialisation.magnitude_initialisation_strategy import MagnitudeInitialisationStrategy
from strategies.initialisation.random_initialisation_strategy import RandomInitialisationStrategy
from strategies.initialisation.snip_initialisation_strategy import SNIPInitialisationStrategy
from strategies.parsing.create_count_func import create_count_func
from strategies.parsing.tokenise_modality import tokenise_modality
from strategies.pruning.magnitude_prune_strategy import MagnitudePruneStrategy
from strategies.pruning.random_prune_strategy import RandomPruneStrategy
from strategies.regrowing.random_normal_xavier_regrow_strategy import RandomNormalXavierRegrowStrategy


def create_dst_strategies(
    modality,
    sparsity,
    weight_counts,
    iterations,
):
    """
    Parse the DST modality string and construct the corresponding
    initialization, pruning, and regrowth strategies.

    Returns:
        (init_strategy, prune_strategy, regrow_strategy)

    Raises:
        ValueError: if any modality or parameter is invalid.
    """

    strategy_sections = tokenise_modality(modality)

    # Expect constant_sparsity v1 format with exactly 3 parts 
    # (the format name, the init part, and the prune part (regrow is implicit from the constant sparsity with normal xavier initialisation))
    if (
        strategy_sections[0][0] != "constant_sparsity"
        or len(strategy_sections[0][1]) != 0
        or len(strategy_sections) != 3
    ):
        raise ValueError(
            f"Invalid modality: {modality}, parsed as {strategy_sections}"
        )

    # Initialisation strategy
    init_mode, init_params = strategy_sections[1]

    # It takes no params
    if len(init_params) != 0:
        raise ValueError(f"Invalid init strategy parameters: {init_mode}, {init_params}")

    # Possible strategies for init
    init_strategies = {
        "dense": None,
        "random": RandomInitialisationStrategy(sparsity),
        "magnitude": MagnitudeInitialisationStrategy(sparsity),
        "grasp": GraSPInitialisationStrategy(sparsity),
        "snip": SNIPInitialisationStrategy(sparsity),
    }

    # Check the init mode is implemented/exists
    if init_mode not in init_strategies:
        raise ValueError(f"Invalid init strategy mode: {init_mode}")

    # init strategy set
    generator_init_strategy = init_strategies[init_mode]

    # create the prune strategy (last thing we will do before returning) (regrow strat is implicitly created from it)
    prune_mode, prune_params = strategy_sections[2]

    # Expect either "static[]" (no dynamic training) or 6 params ["period", int, "fraction", int percentage, "decay", constant or cosine] + a mode
    valid_static = prune_mode == "static" and len(prune_params) == 0
    valid_dynamic = (
        len(prune_params) == 6
        and prune_params[0] == "period"
        and prune_params[2] == "fraction"
        and prune_params[4] == "decay"
    )

    if not (valid_static or valid_dynamic):
        raise ValueError(f"Invalid prune strategy parameters: {prune_mode}, {prune_params}")

    # If no dynamic training is to be done, no prune and regrow strategy
    if prune_mode == "static":
        generator_prune_strategy = None
        generator_regrow_strategy = None
    # Else create the prune strategy, and a regrow strategy to grow back the same amount (constant sparsity) with normal xavier init
    else:
        # Parse parameters
        generator_prune_period = int(prune_params[1])
        generator_prune_fraction = (float(prune_params[3]) / 100) * sparsity
        generator_prune_decay = prune_params[5]

        decay_funcs = {
            "constant": lambda p: generator_prune_fraction if p != 0 and p % generator_prune_period == 0 else None,
            "cosine": lambda p: generator_prune_fraction * np.cos((np.pi * p) / (iterations * 2)) if p != 0 and p % generator_prune_period == 0 else None,
        }

        if generator_prune_decay not in decay_funcs:
            raise ValueError(f"Invalid prune decay type: {generator_prune_decay}")

        generator_fraction_func = decay_funcs[generator_prune_decay]
        generator_count_func = create_count_func(
            generator_fraction_func, generator_prune_period, weight_counts
        )

        prune_strategies = {
            "random": RandomPruneStrategy(generator_count_func),
            "magnitude": MagnitudePruneStrategy(generator_count_func),
        }

        if prune_mode not in prune_strategies:
            raise ValueError(f"Invalid prune mode: {prune_mode}")

        generator_prune_strategy = prune_strategies[prune_mode]
        generator_regrow_strategy = RandomNormalXavierRegrowStrategy(generator_count_func)

    return generator_init_strategy, generator_prune_strategy, generator_regrow_strategy