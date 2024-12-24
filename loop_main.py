'''Loop for the main function.'''

from __future__ import absolute_import, division, print_function

import argparse

from main import main
from utils import load_imputed_data


def get_runs(datasets=None, miss_rates=None, methods=None, inits=None, sparsities=None, n_runs=5, include=None,
             exclude=None):
    # The runs to execute
    if datasets is None:
        datasets = ['letter', 'spam']
    if miss_rates is None:
        miss_rates = [x / 10 for x in range(1, 10)]
    if methods is None:
        methods = ['gain']
    if inits is None:
        inits = ['dense', 'random', 'ER', 'ERK', 'SNIP', 'RSensitivity']
    if sparsities is None:
        sparsities = [x / 10 for x in range(1, 10)]

    runs = {}
    for method in methods:
        if method == 'gain':
            runs.update({
                (dataset, miss_rate, 'gain', init, 0. if init in ('xavier', 'dense', 'full') else sparsity): n_runs
                for dataset in datasets
                for init in inits
                for miss_rate in miss_rates
                for sparsity in sparsities
            })
        else:  # Iterative Imputers
            runs.update({
                (dataset, miss_rate, method, '', 0.): n_runs
                for dataset in datasets
                for miss_rate in miss_rates
            })

    if include is not None: runs.update({
        key: n_runs for key in include
    })

    # Remove completed runs from runs
    imputed_data = load_imputed_data()
    for data in imputed_data:
        # Format the keys
        method = data[2]
        if method == 'IterativeImputer':
            key = (data[0], float(data[1]), 'iterative_imputer', '', float(data[4]))
        elif method == 'IterativeImputerRF':
            key = (data[0], float(data[1]), 'iterative_imputer_rf', '', float(data[4]))
        else:  # GAIN
            key = (data[0], float(data[1]), 'gain', data[3], float(data[4]))

        # Remove runs
        if key in runs:
            runs_left = n_runs - (len(data[4:]))
            if runs_left <= 0:
                runs.pop(key)
            else:
                runs[key] = runs_left

    if exclude is not None:
        for key in exclude:
            if key in runs: runs.pop(key)

    return runs


if __name__ == '__main__':
    # Set the parameters
    datasets_run = ['letter']  # ['spam', 'letter', 'fashion_mnist']
    miss_rates_run = [0.2]
    methods_run = ['gain', 'iterative_imputer']
    inits_run = ['dense', 'random', 'ER']  # GAIN only
    sparsities_run = [0.95]  # Sparse GAIN only
    n_runs_run = 3

    # Exclusions
    exclude = [
        #     (dataset, miss_rate, 'gain', sparsity, 'ER')
        #     for dataset in datasets_run
        #     for miss_rate in miss_rates_run
        #     for sparsity in [0.6, 0.8, 0.9, 0.95]
    ]

    # Execute the runs
    runs = get_runs(datasets_run, miss_rates_run, methods_run, inits_run, sparsities_run, n_runs_run, exclude=exclude)
    while len(runs) > 0:
        for run in runs:
            for x in range(runs[run]):
                # Inputs for the main function
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    '--data_name',
                    default=run[0],
                    type=str)
                parser.add_argument(
                    '--miss_rate',
                    default=run[1],
                    type=float)
                parser.add_argument(
                    '--batch_size',
                    default=128,
                    type=int)
                parser.add_argument(
                    '--hint_rate',
                    default=0.9,
                    type=float)
                parser.add_argument(
                    '--alpha',
                    default=100,
                    type=float)
                parser.add_argument(
                    '--iterations',
                    default=10000,
                    type=int)
                parser.add_argument(
                    '--method',
                    default=run[2],
                    type=str)
                parser.add_argument(
                    '--init',
                    default=run[3],
                    type=str)
                parser.add_argument(
                    '--sparsity',
                    default=run[4],
                    type=float)
                parser.add_argument(
                    "--save",
                    default=True,
                    type=bool)
                parser.add_argument(
                    '--folder',
                    default='imputed_data',
                    type=str)

                args = parser.parse_args()

                # Calls main function
                imputed_data, rmse = main(args, True)

        # Update runs
        runs = get_runs(datasets_run, miss_rates_run, methods_run, inits_run, sparsities_run, n_runs_run,
                        exclude=exclude)
