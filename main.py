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

"""Main function for S-GAIN."""

import argparse
import os

import numpy as np

from models.s_gain_TFv1_FP32 import s_gain
from monitors.monitor import Monitor
from utils.data_loader import data_loader
from utils.load_store import get_filepaths, save_imputation
from utils.metrics import get_rmse


def main(args):
    """Main function for S-GAIN:
    1. Parse the arguments
    2. Load and introduce missing elements in the data according to the provided miss rate and modality
    3. Call S-GAIN
    4. Save the imputed data, logs and trained model and plot the graphs

    :param args:
    - dataset: the dataset to use
    - miss_rate: the probability of missing elements in the data
    - miss_modality: the modality of missing data (MCAR, MAR, MNAR)
    - seed: the seed used to introduce missing elements in the data (optional)
    - batch_size: the number of samples in mini-batch
    - hint_rate: the hint probability
    - alpha: the hyperparameter
    - iterations (epochs): the number of training iterations (epochs)
    - generator_sparsity: the probability of sparsity in the generator
    - generator_modality: the initialization and pruning and regrowth strategy of the generator
    - discriminator_sparsity: the probability of sparsity in the discriminator
    - discriminator_modality: the initialization and pruning and regrowth strategy of the discriminator
    - folder (directory): the folder to save the imputed data to
    - verbose: enable verbose output to console
    - no_log: turn off the logging of metrics (also disables graphs)
    - no_graph: don't plot graphs after training
    - no_model: don't save the trained model
    - no_save: don't save the imputation
    - no_system_information: don't log system information

    :return:
    - imputed_data_x: the imputed data
    - rmse: the root mean squared error
    """

    # Get the parameters
    dataset = args.dataset.lower()
    miss_rate = args.miss_rate
    miss_modality = args.miss_modality.upper()
    seed = args.seed
    batch_size = args.batch_size
    hint_rate = args.hint_rate
    alpha = args.alpha
    iterations = args.iterations
    generator_sparsity = args.generator_sparsity
    generator_modality = args.generator_modality.lower()

    generator_regrowth_strategy = args.generator_regrow_strategy.lower()
    generator_regrowth_period = args.generator_regrow_period
    generator_regrowth_fraction = args.generator_regrow_fraction
    generator_regrowth_decay = args.generator_regrow_decay.lower()

    discriminator_sparsity = args.discriminator_sparsity
    discriminator_modality = args.discriminator_modality.lower()
    folder = args.folder
    verbose = args.verbose
    no_log = args.no_log
    no_graph = args.no_graph
    no_model = args.no_model
    no_save = args.no_save
    no_system_information = args.no_system_information

    # Standardization
    def sparsity_modality(sparsity, modality):
        if sparsity == 0 or modality == 'dense':
            return 0, 'dense'
        elif modality == 'random':
            return sparsity, modality
        elif modality == 'magnitude':
            return sparsity, modality
        elif modality in ('er', 'erdos_renyi'):
            return sparsity, 'ER'
        elif modality in ('erk', 'erdos_renyi_kernel'):
            return sparsity, 'ERK'
        elif modality in ('errw', 'erdos_renyi_random_weight'):
            return sparsity, 'ERRW'
        elif modality in ('erkrw', 'erdos_renyi_kernel_random_weight'):
            return sparsity, 'ERKRW'
        elif modality in ('grasp'):
            return sparsity, 'GraSP'
        elif modality in ('snip'):
            return sparsity, 'SNIP'
        return None

    generator_sparsity, generator_modality = sparsity_modality(generator_sparsity, generator_modality)
    discriminator_sparsity, discriminator_modality = sparsity_modality(discriminator_sparsity, discriminator_modality)

    if seed is None: seed = np.random.randint(2 ** 31)

    # Exit program if a modality is not implemented yet Todo: implement the modalities
    not_implemented = ['MAR', 'MNAR', 'ERK', 'erdos_renyi_kernel', 'ERKRW', 'erdos_renyi_kernel_random_weight',
                       'RSensitivity']
    if miss_modality in not_implemented:
        print(f'Miss modality {miss_modality} is not implemented. Exiting program...')
        return None
    if generator_modality in not_implemented:
        print(f'Generator modality {discriminator_modality} is not implemented. Exiting program...')
        return None
    if discriminator_modality in not_implemented:
        print(f'Discriminator modality {discriminator_modality} is not implemented. Exiting program...')
        return None

    # Name the experiment
    # NOTE this is used by log_and_graphs.py to parse the parameters of the experiment
    experiment = f'S-GAIN_{dataset}_MR_{miss_rate}_MM_{miss_modality}_S_0x{seed:08x}_BS_{batch_size}_HR_{hint_rate}' \
                 f'_a_{alpha}_i_{iterations}_GS_{generator_sparsity}_GM_{generator_modality}' \
                 f'_GRS_{generator_regrowth_strategy}_GRF_{generator_regrowth_fraction}_GRP_{generator_regrowth_period}_GRD_{generator_regrowth_decay}' \
                 f'_DS_{discriminator_sparsity}_DM_{discriminator_modality}'

    if verbose:
        print(experiment)
        print('Loading data...')

    # Load the data with missing elements
    data_x, miss_data_x, data_mask = data_loader(dataset, miss_rate, miss_modality, seed)

    # S-GAIN
    monitor = None if no_log and no_model else Monitor(data_x, data_mask, experiment=experiment, verbose=verbose)
    imputed_data_x = s_gain(
        miss_data_x, batch_size=batch_size, hint_rate=hint_rate, alpha=alpha, iterations=iterations,
        generator_sparsity=generator_sparsity, generator_modality=generator_modality,
        discriminator_sparsity=discriminator_sparsity, discriminator_modality=discriminator_modality,
        verbose=verbose, no_model=no_model, monitor=monitor,
        generator_regrowth_strategy=generator_regrowth_strategy, generator_regrowth_fraction=generator_regrowth_fraction, generator_regrowth_period=generator_regrowth_period, generator_regrowth_decay=generator_regrowth_decay
    )

    # Calculate the RMSE
    rmse = get_rmse(data_x, imputed_data_x, data_mask, round=True)
    if verbose: print(f'RMSE: {rmse}')

    # Save the imputation, the logs and the (trained) model, and plot the graphs
    filepath_imputed_data, filepath_log, filepath_model, filepath_graphs = get_filepaths(folder, experiment, rmse)

    if not no_save:
        if verbose: print('Saving imputation...')
        save_imputation(filepath_imputed_data, imputed_data_x)

    # We need to save stuff with the monitor so flush the logs. We need to do this before running logs_and_graphs, so we need an explicit flush
    # If running this with --no-log, log_and_graphs is never run from this process, and this process exits, the files are flusehd by the
    # python runtime/OS exiting, and running log_and_graphs.py will be fine
    # But if we run this without --no-log, files will not necessarily be up to date, which log_and_graphs requires.
    # This specifically happened for the G_loss.bin file, causing an out of index exception in log_and_graphs.
    monitor.flush_logs()

    if not no_log:
        if no_graph:

            # log_and_graphs.py expects a file "temp/rundata" where its lines are various  positional parameters
            # for logs and graphs
            with open('temp/run_data', 'w') as f:
                f.write(f'{experiment}\n{filepath_imputed_data}\n{filepath_log}\n{"no graphs please?"}\n{filepath_model}')
                f.close()

            os.system(f'python log_and_graphs.py -ng'
                      f'{" -nsi" if no_system_information else ""}'
                      f'{" -v" if verbose else ""}')
        else:

            # log_and_graphs.py expects a file "temp/rundata" where its lines are various  positional parameters
            # for logs and graphs
            with open('temp/run_data', 'w') as f:
                f.write(f'{experiment}\n{filepath_imputed_data}\n{filepath_log}\n{filepath_graphs}\n{filepath_model}')
                f.close()
            
            os.system(f'python log_and_graphs.py'
                      f'{" -nsi" if no_system_information else ""}'
                      f'{" -v" if verbose else ""}')

    else:  # Store data to run log_and_graphs.py later
        with open('temp/run_data', 'w') as f:
            f.write(f'{experiment}\n{filepath_imputed_data}\n{filepath_log}\n{filepath_graphs}\n{filepath_model}')
            f.close()
            

    if not no_model:
        if verbose: print('Saving (trained) model...')
        monitor.save_model(filepath_model)

    if verbose: print(f'Finished.')

    return imputed_data_x, rmse


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='which dataset to use',
        choices=['health', 'letter', 'spam', 'mnist', 'fashion_mnist', 'cifar10'],
        type=str)
    parser.add_argument(
        '-mr', '--miss_rate',
        help='the probability of missing elements in the data',
        default=0.2,
        type=float)
    parser.add_argument(
        '-mm', '--miss_modality',
        help='the modality of missing data (MCAR, MAR, MNAR)',
        choices=['MCAR', 'MAR', 'MNAR'],
        default='MCAR',
        type=str)
    parser.add_argument(
        '-s', '--seed',
        help='the seed used to introduce missing elements in the data (optional)',
        default=None,
        type=int)
    parser.add_argument(
        '-bs', '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '-hr', '--hint_rate',
        help='the hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '-a', '--alpha',
        help='the hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '-i', '--iterations', '-e', '-epochs',
        help='the number of training iterations (epochs)',
        default=10000,
        type=int)
    parser.add_argument(
        '-gs', '--generator_sparsity',
        help='the probability of sparsity in the generator',
        default=0,
        type=float)
    parser.add_argument(
        '-gm', '--generator_modality',
        help='the static initialization strategy of the generator',
        choices=['dense', 'random', 'ER', 'erdos_renyi', 'ERK', 'erdos_renyi_kernel', 'ERRW',
                 'erdos_renyi_random_weight', 'ERKRW', 'erdos_renyi_kernel_random_weight', 'SNIP', 'GraSP',
                 'RSensitivity', 'magnitude'],
        default='dense',
        type=str)
    parser.add_argument(
        '-grs', '--generator_regrow_strategy',
        help='the dynamic regrow/pruning strategy of the generator',
        choices=['static', 'random', 'magnitude', 'grasp', 'magnitude'],
        default='static'
    )
    parser.add_argument(
        '-grf', '--generator_regrow_fraction',
        help='the fraction of weights to regrow for dynamic regrowth for the generator',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '-grp', '--generator_regrow_period',
        help='the dynamic regrow/pruning period for the generator',
        type=int,
        default=200
    )
    parser.add_argument(
        '-grd', '--generator_regrow_decay',
        help='the dynamic regrow/pruning decay the generator',
        choices=['none', 'cosine'],
        default='cosine'
    )
    parser.add_argument(
        '-ds', '--discriminator_sparsity',
        help='thee probability of sparsity in the discriminator',
        default=0,
        type=float)
    parser.add_argument(
        '-dm', '--discriminator_modality',
        help='the initialization and pruning and regrowth strategy of the discriminator',
        choices=['dense', 'random', 'ER', 'erdos_renyi', 'ERK', 'erdos_renyi_kernel', 'ERRW',
                 'erdos_renyi_random_weight', 'ERKRW', 'erdos_renyi_kernel_random_weight', 'SNIP', 'GraSP',
                 'RSensitivity'],
        default='dense',
        type=str)
    parser.add_argument(
        '-f', '--folder', '-d', '-dir', '--directory',
        help='save the imputed data to a different folder (optional)',
        default='output',
        type=str)
    parser.add_argument(
        '-v', '--verbose',
        help='enable verbose logging',
        action='store_true')
    parser.add_argument(
        '-nl', '--no_log',
        help='turn off the logging of metrics (also disables graphs)',
        action='store_true')
    parser.add_argument(
        '-ng', '--no_graph',
        help="don't plot graphs after training",
        action='store_true')
    parser.add_argument(
        '-nm', '--no_model',
        help="don't save the trained model",
        action='store_true')
    parser.add_argument(
        '-ns', '--no_save',
        help="don't save the imputation",
        action='store_true')
    parser.add_argument(
        '-nsi', '--no_system_information',
        help="don't log system information",
        action='store_true')
    args = parser.parse_args()

    main(args)
