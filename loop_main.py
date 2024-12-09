'''Loop for the main function.'''

from __future__ import absolute_import, division, print_function

import argparse

from main import main
from utils import load_imputed_data


def get_runs(datasets=None, miss_rates=None, sparsities=None, inits=None, n_runs=5):
    # The runs to execute
    if datasets is None:
        datasets = ['letter', 'spam']
    if miss_rates is None:
        miss_rates = [x / 10 for x in range(1, 10)]
    if sparsities is None:
        sparsities = [x / 10 for x in range(1, 10)]
    if inits is None:
        inits = ['dense', 'random', 'ER', 'ERK', 'SNIP', 'RSensitivity']

    runs = {
        (dataset, miss_rate, 0 if init in ('xavier', 'dense', 'full') else sparsity, init): n_runs
        for dataset in datasets
        for init in inits
        for miss_rate in miss_rates
        for sparsity in sparsities
    }

    # Remove completed runs from runs
    imputed_data = load_imputed_data()
    for data in imputed_data:
        key = (data[0], float(data[1]), float(data[2]), data[3])
        if key in runs:
            runs_left = n_runs - (len(data[4:]))
            if runs_left <= 0:
                runs.pop(key)
            else:
                runs[key] = runs_left

    return runs


if __name__ == '__main__':
    # Set the parameters
    datasets_run = ['spam', 'letter']
    miss_rates_run = [0.2]
    sparsities_run = [0.6, 0.8, 0.9, 0.95, 0.99]
    inits_run = ['dense', 'random', 'ER']
    n_runs_run = 25

    # Execute the runs
    runs = get_runs(datasets_run, miss_rates_run, sparsities_run, inits_run, n_runs_run)
    while len(runs) > 0:
        for run in runs:
            for x in range(runs[run]):
                # Inputs for the main function
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    '--data_name',
                    choices=['letter', 'spam'],
                    default=run[0],
                    type=str)
                parser.add_argument(
                    '--miss_rate',
                    help='missing data probability',
                    default=run[1],
                    type=float)
                parser.add_argument(
                    '--batch_size',
                    help='the number of samples in mini-batch',
                    default=128,
                    type=int)
                parser.add_argument(
                    '--hint_rate',
                    help='hint probability',
                    default=0.9,
                    type=float)
                parser.add_argument(
                    '--alpha',
                    help='hyperparameter',
                    default=100,
                    type=float)
                parser.add_argument(
                    '--iterations',
                    help='number of training iterations',
                    default=10000,
                    type=int)
                parser.add_argument(
                    '--sparsity',
                    help='probability of sparsity in the generator',
                    default=run[2],
                    type=float)
                parser.add_argument(
                    '--init',
                    choices=['xavier', 'dense', 'full', 'random', 'erdos_renyi', 'er', 'snip', 'rsensitivity'],
                    default=run[3],
                    type=str)
                parser.add_argument(
                    "--save",
                    help='save the output to csv file',
                    default=True,
                    type=bool)
                parser.add_argument(
                    '--folder',
                    help='the folder to save the csv files in',
                    default='imputed_data',
                    type=str)

                args = parser.parse_args()

                # Calls main function
                imputed_data, rmse = main(args, True)

        # Update runs
        runs = get_runs(datasets_run, miss_rates_run, sparsities_run, inits_run, n_runs_run)
