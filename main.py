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

'''Main function for UCI letter and spam datasets.'''

# Necessary packages
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np

from data_loader import data_loader
from utils import rmse_loss, save_imputation_results
from gain import gain
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def iterative_imputer(miss_data_x, seed=None):
    """Perform missing data imputation using Iterative Imputer."""
    if seed is not None:
        imputer = IterativeImputer(max_iter=10, random_state=seed)
    else:
        imputer = IterativeImputer(max_iter=10)

    imputed_data_x = imputer.fit_transform(miss_data_x)
    return imputed_data_x


def iterative_imputer_rf(miss_data_x, seed=None):
    """Perform missing data imputation using Iterative Imputer with RandomForest."""
    if seed is not None:
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=seed)
        imputer = IterativeImputer(estimator=rf_regressor, max_iter=10, random_state=seed)
    else:
        rf_regressor = RandomForestRegressor(n_estimators=100)
        imputer = IterativeImputer(estimator=rf_regressor, max_iter=10)

    imputed_data_x = imputer.fit_transform(miss_data_x)
    return imputed_data_x


def main(args, loop=False):
    '''Main function for UCI letter and spam datasets.

    Args:
      - data_name: letter or spam
      - miss_rate: probability of missing components
      - batch_size: batch size
      - hint_rate: hint rate
      - alpha: hyperparameter
      - iterations: iterations
      - method: which method to use (gain, iterative_imputer, iterative_imputer_rf)
      - sparsity: probability of sparsity in the generator (GAIN only)
      - init: which initialization to use (xavier, random, erdos_renyi, snip, rsensitivity) (GAIN only)
      - save: save the output to csv file
      - folder: the folder to save the csv files to

    Returns:
      - imputed_data_x: imputed data
      - rmse: Root Mean Squared Error
    '''

    data_name = args.data_name
    miss_rate = args.miss_rate
    save = args.save
    folder = args.folder

    if data_name == 'fashion_mnist':
        data_name = 'FashionMNIST'
    elif data_name == 'mnist':
        data_name = 'MNIST'

    if args.method.lower() == 'gain':
        method = 'GAIN'
        if args.init.lower() == 'random':
            args.init = 'random'
        elif args.init.lower() in ('erdos_renyi', 'er'):
            args.init = 'ER'
        elif args.init.lower() in ('erdos_renyi_kernel', 'erk'):
            args.init = 'ERK'
        elif args.init.lower() == 'snip':
            args.init = 'SNIP'
        elif args.init.lower() == 'rsensitivity':
            args.init = 'RSensitivity'
        else:
            args.sparsity = 0.
            args.init = 'dense'
    elif args.method.lower() == 'iterative_imputer':
        method = 'IterativeImputer'
        args.sparsity = 0.
        args.init = ''
    elif args.method.lower() == 'iterative_imputer_rf':
        method = 'IterativeImputerRF'
        args.sparsity = 0.
        args.init = ''
    else:
        print("Invalid method selection. Exiting the program.")
        return

    # Print the command if run in a loop
    if loop: print(
        f'python main.py --data_name {data_name} --miss_rate {miss_rate} --batch_size {args.batch_size}'
        f' --hint_rate {args.hint_rate} --alpha {args.alpha} --iterations {args.iterations} --method {method}'
        f' --sparsity {args.sparsity} --init {args.init} --save {save} --folder {folder}'
    )

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
    if method == 'GAIN':
        print('Impute missing data using GAIN')
        gain_parameters = {
            'batch_size': args.batch_size,
            'hint_rate': args.hint_rate,
            'alpha': args.alpha,
            'iterations': args.iterations,
            'sparsity': args.sparsity,
            'init': args.init
        }
        imputed_data_x, G_tensors = gain(miss_data_x, gain_parameters)
    elif method == 'IterativeImputer':
        print('Impute missing data using Iterative Imputer')
        imputed_data_x = iterative_imputer(miss_data_x)
        G_tensors = []  # No tensors to save
    else:  # IterativeImputerRF
        print('Impute using Iterative Imputer with RandomForest Regressor')
        imputed_data_x = iterative_imputer_rf(miss_data_x)
        G_tensors = []  # No tensors to save

    # Report the RMSE performance
    rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
    rsme_performance = str(np.round(rmse, 4))
    print(f'\n{method} {args.init} RMSE Performance: ' + rsme_performance)

    # Save the imputed data
    if save: save_imputation_results(imputed_data_x, data_name, miss_rate, method, args.init, args.sparsity,
                                     rsme_performance, G_tensors, folder)

    return imputed_data_x, rmse


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['letter', 'spam', 'mnist', 'fashion_mnist'],
        default='mnist',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
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
        '--method',
        choices=['gain', 'iterative_imputer', 'iterative_imputer_rf'],
        default='gain',
        type=str)
    parser.add_argument(
        '--init',
        help='initialization method (GAIN only)',
        choices=['xavier', 'dense', 'full', 'random', 'erdos_renyi', 'er', 'snip', 'rsensitivity'],
        default='full',
        type=str)
    parser.add_argument(
        '--sparsity',
        help='probability of sparsity in the generator (GAIN only)',
        default=0,
        type=float)
    parser.add_argument(
        '--save',
        help='save the output to csv file',
        action="store_true",
        default=False,
        type=bool)
    parser.add_argument(
        '--folder',
        help='the folder to save the csv files in',
        default='imputed_data',
        type=str)

    args = parser.parse_args()

    # Calls main function
    imputed_data, rmse = main(args)
