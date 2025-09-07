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

"""Load and store operations for S-GAIN:

(1) get_filepaths: create the necessary directory and return the appropriate filepaths
(2) save_imputation: save the imputed data to a csv file
(3) parse_experiment: parse the experiment
(4) parse_files: parse the output files to a pandas dataframe
(5) parse_log: parse the log file to a list
(6) get_experiments: get a dictionary (or a list of strings) of the experiments to run
(7) read_bin: read a (temporary) binary file
"""

import json
import struct

import pandas as pd

from os import makedirs, listdir
from os.path import isdir, isfile


def get_filepaths(directory, experiment, rmse):
    """Create the necessary directory and return the appropriate filepaths

    :param directory: the directory to save to
    :param experiment: the name of the experiment
    :param rmse: the Root Mean Squared Error

    :return:
    - filepath_imputed_data: the filepath for the imputed data
    - filepath_log: the filepath for the log
    - filepath_model: the filepath for the (trained) model
    - filepath_rmse: the filepath for the RMSE graph
    - filepath_imputation_time: the filepath for the imputation time graph
    - filepath_memory_usage: the filepath for the memory usage graph
    - filepath_energy_consumption: the filepath for the energy consumption graph
    - filepath_sparsity: the filepath for the sparsity graph
    - filepath_flops: the filepath for the FLOPs graph
    - filepath_loss: the filepath for the loss graph
    """

    if not isdir(directory): makedirs(directory)
    temp_filepath = f'{directory}/{experiment}_RMSE_{rmse}'

    # Avoid overwriting if RMSE is the same
    if (isfile(f'{temp_filepath}.csv')
            or isfile(f'{temp_filepath}_log.json')
            or isfile(f'{temp_filepath}_model.json')
            or isfile(f'{temp_filepath}_graphs.png')
            or isfile(f'{temp_filepath}_RMSE.png')
            or isfile(f'{temp_filepath}_imputation_time.png')
            or isfile(f'{temp_filepath}_memory_usage.png')
            or isfile(f'{temp_filepath}_energy_consumption.png')
            or isfile(f'{temp_filepath}_sparsity.png')
            or isfile(f'{temp_filepath}_FLOPs.png')
            or isfile(f'{temp_filepath}_loss.png')
    ):
        i = 1
        while (isfile(f'{temp_filepath}_{i}.csv')
               or isfile(f'{temp_filepath}_{i}_log.json')
               or isfile(f'{temp_filepath}_{i}_model.json')
               or isfile(f'{temp_filepath}_{i}_graphs.png')
               or isfile(f'{temp_filepath}_{i}_RMSE.png')
               or isfile(f'{temp_filepath}_{i}_imputation_time.png')
               or isfile(f'{temp_filepath}_{i}_memory_usage.png')
               or isfile(f'{temp_filepath}_{i}_energy_consumption.png')
               or isfile(f'{temp_filepath}_{i}_sparsity.png')
               or isfile(f'{temp_filepath}_{i}_FLOPs.png')
               or isfile(f'{temp_filepath}_{i}_loss.png')
        ): i += 1
        temp_filepath = f'{temp_filepath}_{i}'

    filepath_imputed_data = f'{temp_filepath}.csv'
    filepath_log = f'{temp_filepath}_log.json'
    filepath_model = f'{temp_filepath}_model.json'
    filepath_graphs = f'{temp_filepath}_graphs.png'
    filepath_rmse = f'{temp_filepath}_RMSE.png'
    filepath_imputation_time = f'{temp_filepath}_imputation_time.png'
    filepath_memory_usage = f'{temp_filepath}_memory_usage.png'
    filepath_energy_consumption = f'{temp_filepath}_energy_consumption.png'
    filepath_sparsity = f'{temp_filepath}_sparsity.png'
    filepath_flops = f'{temp_filepath}_FLOPs.png'
    filepath_loss = f'{temp_filepath}_loss.png'

    return filepath_imputed_data, filepath_log, filepath_model, filepath_graphs, filepath_rmse, \
        filepath_imputation_time, filepath_memory_usage, filepath_energy_consumption, filepath_sparsity, \
        filepath_flops, filepath_loss


def save_imputation(filepath, imputed_data_x):
    """Save the imputed data to a CSV file.

    :param filepath: the filepath to save to
    :param imputed_data_x: the imputed data
    """

    # Save the imputation
    imputed_data_x = pd.DataFrame(imputed_data_x)
    imputed_data_x.to_csv(filepath, index=False)


def parse_experiment(experiment, file=False):
    """Parse the experiment.

    :param experiment: the name of the experiment
    :param file: whether parsing a file or not

    :return:
    - False: if the experiment is not in S-GAIN format
    - dataset: the dataset used
    - miss_rate: the probability of missing elements in the data
    - miss_modality: the modality of missing data (MCAR, MAR, MNAR)
    - seed: the seed used to introduce missing elements in the data
    - batch_size: the number of samples in mini-batch
    - hint_rate: the hint probability
    - alpha: the hyperparameter
    - iterations (epochs): the number of training iterations (epochs)
    - generator_sparsity: the probability of sparsity in the generator
    - generator_modality: the initialization and pruning and regrowth strategy of the generator
    - discriminator_sparsity: the probability of sparsity in the discriminator
    - discriminator_modality: the initialization and pruning and regrowth strategy of the discriminator
    - rmse: the RMSE (if parsing a file)
    - index: the index of the experiment (if parsing a file)
    - filetype: the type of file (imputed_data, log, model, etc.)
    """

    # Check if the experiment belongs to S-GAIN
    if not experiment.startswith('S-GAIN'): return False

    # Remove the file extension
    if file: experiment = experiment.rsplit('.', 1)[0]

    # Parse experiment
    _, rest = experiment.split('S-GAIN_')
    dataset, rest = rest.split('_MR_')
    miss_rate, rest = rest.split('_MM_')
    miss_rate = float(miss_rate)
    miss_modality, rest = rest.split('_S_')
    seed, rest = rest.split('_BS_')
    seed = int(seed, 16)
    batch_size, rest = rest.split('_HR_')
    batch_size = int(batch_size)
    hint_rate, rest = rest.split('_a_')
    hint_rate = float(hint_rate)
    alpha, rest = rest.split('_i_')
    alpha = float(alpha)
    iterations, rest = rest.split('_GS_')
    iterations = int(iterations)
    generator_sparsity, rest = rest.split('_GM_')
    generator_sparsity = float(generator_sparsity)
    generator_modality, rest = rest.split('_DS_')
    discriminator_sparsity, rest = rest.split('_DM_')
    discriminator_sparsity = float(discriminator_sparsity)

    if file:  # rmse, index, filetype
        discriminator_modality, rest = rest.split('_RMSE_')

        rests = rest.split('_', 1)
        rmse = float(rests[0])

        if len(rests) == 1:
            index = 0
            filetype = 'imputed_data'
        else:  # index, filetype
            index_filetype = rests[1].split('_', 1)
            if len(index_filetype) == 1:  # index or filetype
                if index_filetype[0].isdigit():  # index
                    index = int(index_filetype[0])
                    filetype = 'imputed_data'
                else:  # filetype
                    index = 0
                    filetype = index_filetype[0]
            elif index_filetype[0].isdigit():  # index and filetype
                index = int(index_filetype[0])
                filetype = index_filetype[1]
            else:  # filetype
                index = 0
                filetype = rests[1]

        return dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity, \
            generator_modality, discriminator_sparsity, discriminator_modality, rmse, index, filetype
    else:
        discriminator_modality = rest

        return dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, \
            generator_sparsity, generator_modality, discriminator_sparsity, discriminator_modality


def parse_files(files=None, filepath='output', filetype=None):
    """Parse the output files to a pandas dataframe.

    :param files: a list of files to parse (optional)
    :param filepath: the output filepath
    :param filetype: only return files of this type (imputed_data, log, model, etc.)

    :return:
    - df_files: a Pandas DataFrame with all the experiments
    """

    files = [parse_experiment(file, file=True) for file in files] if files \
        else [parse_experiment(file, file=True) for file in listdir(filepath)] if isdir(filepath) \
        else []

    header = ['dataset', 'miss_rate', 'miss_modality', 'seed', 'batch_size', 'hint_rate', 'alpha', 'iterations',
              'generator_sparsity', 'generator_modality', 'discriminator_sparsity', 'discriminator_modality', 'rmse',
              'index', 'filetype']

    if filetype is not None: files = [file for file in files if file[-1] == filetype]

    df_files = pd.DataFrame(files, columns=header)
    return df_files


def parse_log(filepath_log):
    """Parse the log file to a list.

    :param filepath_log: the filepath for the log file

    :return:
    - RMSE: the RMSE log
    - imputation_time: the imputation time log
    - memory_usage: the memory usage log
    - energy_consumption: the energy consumption log
    - sparsity: the sparsity log (total)
    - sparsity_G: the sparsity log for the generator
    - sparsity_G_W1: the sparsity log for the first layer of the generator
    - sparsity_G_W2: the sparsity log for the second layer of the generator
    - sparsity_G_W3: the sparsity log for the third layer of the generator
    - sparsity_D: the sparsity log for the discriminator
    - sparsity_D_W1: the sparsity log for the first layer of the discriminator
    - sparsity_D_W2: the sparsity log for the second layer of the discriminator
    - sparsity_D_W3: the sparsity log for the third layer of the discriminator
    - FLOPs: the FLOPs log (total)
    - FLOPs_G: the FLOPs log for the generator
    - FLOPs_D: the FLOPs log for the discriminator
    - loss_G: the loss log for the generator (cross entropy)
    - loss_D: the loss log for the discriminator (cross entropy)
    - loss_MSE: the los log (MSE)
    """

    # Read the log file
    f_log = open(filepath_log, 'r')
    log = json.loads(f_log.read())
    f_log.close()

    # Retrieve the logs
    RMSE = log['rmse']['log']
    imputation_time = log['imputation_time']['log']
    memory_usage = log['memory_usage']['log']
    energy_consumption = log['energy_consumption']['log']
    sparsity = log['sparsity']['log']
    sparsity_G = log['sparsity']['generator']['log']
    sparsity_G_W1 = log['sparsity']['generator']['G_W1']['log']
    sparsity_G_W2 = log['sparsity']['generator']['G_W2']['log']
    sparsity_G_W3 = log['sparsity']['generator']['G_W3']['log']
    sparsity_D = log['sparsity']['discriminator']['log']
    sparsity_D_W1 = log['sparsity']['discriminator']['D_W1']['log']
    sparsity_D_W2 = log['sparsity']['discriminator']['D_W2']['log']
    sparsity_D_W3 = log['sparsity']['discriminator']['D_W3']['log']
    FLOPs = log['flops']['log']
    FLOPs_G = log['flops']['generator']['log']
    FLOPs_D = log['flops']['discriminator']['log']
    loss_G = log['loss']['cross_entropy']['generator']['log']
    loss_D = log['loss']['cross_entropy']['discriminator']['log']
    loss_MSE = log['loss']['MSE']['log']

    return RMSE, imputation_time, memory_usage, energy_consumption, sparsity, sparsity_G, sparsity_G_W1, \
        sparsity_G_W2,  sparsity_G_W3, sparsity_D, sparsity_D_W1, sparsity_D_W2, sparsity_D_W3, FLOPs, FLOPs_G, \
        FLOPs_D, loss_G, loss_D, loss_MSE


def get_experiments(datasets, miss_rates=None, miss_modalities=None, seeds=None, batch_sizes=None, hint_rates=None,
                    alphas=None, iterations_s=None, generator_sparsities=None, generator_modalities=None,
                    discriminator_sparsities=None, discriminator_modalities=None, folder='output', n_runs=10,
                    ignore_existing_files=False, retry_failed_experiments=True, include=None, exclude=None,
                    verbose=False, no_log=False, no_graph=False, no_model=False, no_save=False, get_commands=False):
    """Get a dictionary (or a list of strings) of the experiments to run.

    :param datasets: which datasets to use
    :param miss_rates: the probabilities of missing elements in the data
    :param miss_modalities: the modalities of missing data (MCAR, MAR, MNAR)
    :param seeds: the seeds used to introduce missing elements in the data (optional)
    :param batch_sizes: a list of the number of samples in mini-batch
    :param hint_rates: the hint probabilities
    :param alphas: the hyperparameters
    :param iterations_s: a list of the number of training iterations (epochs)
    :param generator_sparsities: the probabilities of sparsity in the generator
    :param generator_modalities: the initializations and pruning and regrowth strategies of the generator
    :param discriminator_sparsities: the probabilities of sparsity in the discriminator
    :param discriminator_modalities: the initializations and pruning and regrowth strategies of the discriminator
    :param folder: the output folder (change if saving to a different folder)
    :param n_runs: the number of times each experiment should run (default: 10)
    :param ignore_existing_files: ignore existing files in the output folder (always run the experiment n times)
    :param retry_failed_experiments: retry failed experiments (default: True)
    :param include: include these keys in the experiments to run
    :param exclude: exclude these keys from the experiments to run
    :param verbose: enable verbose output to console
    :param no_log: turn off the logging of metrics (also disables graphs and model)
    :param no_graph: don't plot graphs after training
    :param no_model: don't save the trained model
    :param no_save: don't save the imputation
    :param get_commands: get a list of ready to run commands instead of a dictionary

    :return:
    - experiments: a dictionary (or a list of strings) of the experiments to run
    """

    # Default values (as used in the Sparse GAIN paper)
    if miss_rates is None: miss_rates = [0.2]
    if miss_modalities is None: miss_modalities = ['MCAR']
    if seeds is None: seeds = [0]
    if batch_sizes is None: batch_sizes = [128]
    if hint_rates is None: hint_rates = [0.9]
    if alphas is None: alphas = [100]
    if iterations_s is None: iterations_s = [10000]
    if generator_sparsities is None: generator_sparsities = [0, 0.6, 0.8, 0.9, 0.95, 0.99]
    if generator_modalities is None: generator_modalities = ['dense', 'random', 'ER', 'ERRW']
    if discriminator_sparsities is None: discriminator_sparsities = [0]
    if discriminator_modalities is None: discriminator_modalities = ['dense']

    # Standardization
    def sparsities_modalities(sparsities, modalities):
        sparsity_modality = [
            (0, 'dense')
            for sparsity in sparsities if sparsity == 0
            for modality in modalities if modality == 'dense'
        ]
        sparsity_modality += [
            (sparsity, 'random')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality == 'random'
        ]
        sparsity_modality += [
            (sparsity, 'ER')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ER', 'erdos_renyi')
        ]
        sparsity_modality += [
            (sparsity, 'ERK')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERK', 'erdos_renyi_kernel')
        ]
        sparsity_modality += [
            (sparsity, 'ERRW')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERRW', 'erdos_renyi_random_weight')
        ]
        sparsity_modality += [
            (sparsity, 'ERKRW')
            for sparsity in sparsities if sparsity > 0
            for modality in modalities if modality in ('ERKRW', 'erdos_renyi_kernel_random_weight')
        ]
        return sparsity_modality

    generator_sparsity_modality = sparsities_modalities(generator_sparsities, generator_modalities)
    discriminator_sparsity_modality = sparsities_modalities(discriminator_sparsities, discriminator_modalities)

    # Get the experiments
    experiments = {}
    experiments.update({
        (dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity,
         generator_modality, discriminator_sparsity, discriminator_modality): n_runs
        for dataset in datasets
        for miss_rate in miss_rates
        for miss_modality in miss_modalities
        for seed in seeds
        for batch_size in batch_sizes
        for hint_rate in hint_rates
        for alpha in alphas
        for iterations in iterations_s
        for generator_sparsity, generator_modality in generator_sparsity_modality
        for discriminator_sparsity, discriminator_modality in discriminator_sparsity_modality
    })

    # Add inclusions
    if include is not None: experiments.update({
        key: n_runs for key in include
    })

    # Remove completed experiments
    if not ignore_existing_files:
        df = parse_files(filepath=folder)
        df.drop('filetype', axis=1, inplace=True)  # ignore filetype
        df.drop_duplicates(inplace=True)  # remove duplicates
        if retry_failed_experiments: df.dropna(inplace=True)  # remove failed experiments from the count
        df.drop(['rmse', 'index'], axis=1, inplace=True)  # ignore rmse and index
        df = df.groupby(df.columns.tolist(), as_index=False).size()  # count
        completed = {tuple(exp[:-1]): exp[-1] for exp in df.values}

        for experiment, n_completed in completed.items():
            if experiment in experiments:
                if experiments[experiment] - n_completed <= 0:
                    experiments.pop(experiment)
                else:
                    experiments[experiment] = n_runs - n_completed

    # Remove exclusions
    if exclude is not None:
        for key in exclude:
            if key in experiments: experiments.pop(key)

    # Convert to executable strings
    if get_commands:
        commands = [
            f'python main.py {dataset} --miss_rate {miss_rate} --miss_modality {miss_modality} --seed {seed} '
            f'--batch_size {batch_size} --hint_rate {hint_rate} --alpha {alpha} --iterations {iterations} '
            f'--generator_sparsity {generator_sparsity} --generator_modality {generator_modality} '
            f'--discriminator_sparsity {discriminator_sparsity} --discriminator_modality {discriminator_modality} '
            f'--folder {folder} {"--verbose " if verbose else ""}{"--no_log " if no_log else ""}'
            f'{"--no_graph " if no_graph else ""}{"--no_model " if no_model else ""}{"--no_save" if no_save else ""}'

            for [dataset, miss_rate, miss_modality, seed, batch_size, hint_rate, alpha, iterations, generator_sparsity,
            generator_modality, discriminator_sparsity, discriminator_modality], n in experiments.items()
            for _ in range(n)
        ]
        return commands

    return experiments


def read_bin(filepath):
    """Read a (temporary) binary file.

    :param filepath: the filepath

    :return: the unpacked data from the file
    """

    # Read binary data
    file = open(filepath, 'rb')
    data = file.read()
    file.close()

    # Unpack the data
    fmt = '<%df' % (len(data) // 4)
    data = list(struct.unpack(fmt, data))

    return data
