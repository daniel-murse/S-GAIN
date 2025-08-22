from __future__ import absolute_import, division, print_function

# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loop for the main function."""

import argparse
import pandas as pd

from main import main
from utils import load_imputed_data


def get_runs(datasets, miss_rates, methods, inits, sparsities, n_nearest_features, n_runs, include=None, exclude=None):
    runs = {}
    for method in methods:
        if method == 'gain':
            runs.update({
                (dataset, miss_rate, 'gain', init, sparsity if init != 'Dense' else 0., None): n_runs
                for dataset in datasets
                for init in inits
                for miss_rate in miss_rates
                for sparsity in sparsities
            })
        elif method in ('expectation_maximization', 'em'):
            runs.update({
                (dataset, miss_rate, 'em', '', 0., None): n_runs
                for dataset in datasets
                for miss_rate in miss_rates
            })
        else:  # Iterative Imputers
            runs.update({
                (dataset, miss_rate, method, '', 0., nnf): n_runs
                for dataset in datasets
                for miss_rate in miss_rates
                for nnf in n_nearest_features
            })

    if include is not None: runs.update({
        key: n_runs for key in include
    })

    # Remove completed runs from runs
    imputed_data = load_imputed_data()
    for data in imputed_data:
        data_name, miss_rate, method, init, sparsity, nnf = data[:6]

        # Standardize data_name
        if data_name == 'FashionMNIST':
            data_name = 'fashion_mnist'
        elif data_name == 'MNIST':
            data_name = 'mnist'
        elif data_name == 'CIFAR10':
            data_name = 'cifar10'

        # Format the keys
        if method == 'IterativeImputer':
            key = (data_name, miss_rate, 'iterative_imputer', '', sparsity, nnf)
        elif method == 'IterativeImputerRF':
            key = (data_name, miss_rate, 'iterative_imputer_rf', '', sparsity, nnf)
        else:  # GAIN
            key = (data_name, miss_rate, 'gain', init, sparsity, nnf)

        # Remove runs
        if key in runs:
            runs_left = n_runs - len(data[6:])
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
    datasets_run = ['spam', 'letter', 'health', 'fashion_mnist']
    miss_rates_run = [0.2]
    methods_run = ['gain', 'iterative_imputer', 'iterative_imputer_rf']
    inits_run = ['Dense', 'Random', 'ER', 'ERRW']  # GAIN only
    sparsities_run = [0.6, 0.8, 0.9, 0.95, 0.99]  # Sparse GAIN only
    n_nearest_features_run = [None]  # Iterative Imputers only
    n_runs_run = 10

    # Inclusions
    include = [
        # (dataset, miss_rate, method, init, sparsity, n_nearest_features)
        ('fashion_mnist', 0.2, 'iterative_imputer', '', 0., 100)
    ]

    # Exclusions
    exclude = [
        # (dataset, miss_rate, method, init, sparsity, n_nearest_features)
        ('fashion_mnist', 0.2, method, '', 0., None)
        for method in ['iterative_imputer', 'iterative_imputer_rf']
    ]
    exclude += [
        ('spam', 0.2, 'gain', 'ER', sparsity, None)
        for sparsity in [0.6, 0.8, 0.9, 0.95]
    ]
    exclude += [
        ('letter', 0.2, 'gain', 'ER', sparsity, None)
        for sparsity in [0.6, 0.8]
    ]
    exclude += [
        ('fashion_mnist', 0.2, 'gain', 'ER', sparsity, None)
        for sparsity in [0.6, 0.8, 0.9, 0.95, 0.99]
    ]

    # Get the runs
    runs = get_runs(datasets_run, miss_rates_run, methods_run, inits_run, sparsities_run, n_nearest_features_run,
                    n_runs_run, include, exclude)

    # Initial report on the progress of the loop
    total_runs = sum(runs.values())
    print(f'Progress loop: 0% completed (0/{total_runs})\n')

    # Execute the runs
    while len(runs) > 0:
        i = 0
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
                    '--n_nearest_features',
                    default=run[5],
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

                # Report on the progress of the loop
                if pd.notna(rmse): i += 1
                print(f'Progress loop: {int(i / total_runs * 100)}% completed ({i}/{total_runs})\n')

        # Update runs
        runs = get_runs(datasets_run, miss_rates_run, methods_run, inits_run, sparsities_run, n_nearest_features_run,
                        n_runs_run, include, exclude)
