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

"""Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalized data
(3) rounding: Handle categorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) random_init: Random initialization
(7) erdos_renyi_init: Erdos Renyi initialization
(8) erdos_renyi_kernel_init: Erdos Renyi kernel initialization
(9) erdos_renyi_random_weights_init: Erdos Renyi with Random Weights initialization
(10) erdos_renyi_kernel_random_weights_init: Erdos Renyi Kernel with Random Weights initialization
(11) snip_init: SNIP initialization
(12) rsensitivity_init: RSensitivity initialization
(13) binary_sampler: sample binary random variables
(14) uniform_sampler: sample uniform random variables
(15) sample_batch_index: sample random batch index
(16) save_imputation_results: Save the imputation and initialized tensors to csv files
(17) load_imputed_data: Load the RMSE scores of the imputed data
(18) get_flops: compute the inference FLOPs
"""

# Necessary packages
from os import makedirs, listdir
from os.path import isdir, isfile
import numpy as np
import pandas as pd
# import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from flops.sparse_utils import get_stats
import keras


def normalization(data, parameters=None):
    """Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    """

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:
        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    """Re-normalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: re-normalized original data
    """

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    """

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def rmse_loss(ori_data, imputed_data, data_m):
    """Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    """

    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse


def xavier_init(size, seed=None):
    """Xavier initialization.

    Args:
      - size: vector size
      - seed: random seed

    Returns:
      - initialized normal random vector.
    """

    # Fix seed for run-to-run consistency
    if seed is not None: tf.random.set_random_seed(seed)

    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def random_init(tensors, sparsity, fix_seed=False):
    """Random initialization.

    Args:
        - tensors: the tensors to apply sparsity on
        - sparsity: the level of sparsity [0,1)
        - fix_seed: fix random seed

    Returns:
        - initialized randomly sparsed tensors.
    """

    for i in range(len(tensors)):
        # Fix seed for run-to-run consistency
        if fix_seed: np.random.seed(i)

        tensor = tensors[i]
        mask = np.random.choice([0, 1], size=tensor.shape, p=[sparsity, 1 - sparsity])
        tensors[i] = tensor * mask
    return tensors


def erdos_renyi_init(tensors, sparsity, erk_power_scale=1.0, fix_seed=False):
    """Erdos Renyi initialization.

    Args:
        - tensors: the tensors to apply sparsity on
        - sparsity: the level of sparsity [0,1)
        - erk_power_scale: ?
        - fix_seed: fix random seed

    Returns:
        - initialized Erdos Renyi sparsed tensors.
    """

    total_params = 0
    for name, weight in tensors.items():
        total_params += weight.size
    is_epsilon_valid = False

    dense_layers = set()
    while not is_epsilon_valid:
        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, mask in tensors.items():
            n_param = np.prod(mask.shape)
            n_zeros = n_param * sparsity
            n_ones = n_param * (1 - sparsity)

            if name in dense_layers:
                rhs -= n_zeros
            else:
                rhs += n_ones
                raw_probabilities[name] = (np.sum(mask.shape[:2]) / np.prod(mask.shape[:2])) ** erk_power_scale
                divisor += raw_probabilities[name] * n_param
        epsilon = rhs / divisor
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    print(f'Sparsity of var:{mask_name} had to be set to 0.')
                    dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    density_dict = {}
    total_nonzero = 0.0
    # With the valid epsilon, we can set sparsities of the remaining layers.
    i = 0
    for name, mask in tensors.items():
        # Fix seed for run-to-run consistency
        if fix_seed is not None: np.random.seed(i)

        if name in dense_layers:
            density_dict[name] = 1.0
        else:
            probability_one = epsilon * raw_probabilities[name]
            density_dict[name] = probability_one
        print(f'layer: {name}, shape: {mask.shape}, sparsity: {1 - density_dict[name]}')
        tensors[name] = tf.convert_to_tensor(np.random.rand(*mask.shape) < density_dict[name], dtype='float32')
        total_nonzero += density_dict[name] * mask.size
        i += 1
    print(f'Overall sparsity {1 - total_nonzero / total_params}')

    return tensors


def erdos_renyi_kernel_init(tensors, sparsity, erk_power_scale=1.0, fix_seed=False):
    """Erdos Renyi Kernel initialization.

    Args:
        - tensors: the tensors to apply sparsity on
        - sparsity: the level of sparsity [0,1)
        - erk_power_scale: ?
        - fix_seed: fix random seed

    Returns:
        - initialized Erdos Renyi Kernel sparsed tensors.
    """

    tensors = erdos_renyi_init(tensors, sparsity, erk_power_scale, fix_seed)

    # for module in self.modules:
    #     for name, tensor in module.named_parameters():
    #         if name in tensors:
    #             tensor.data = tensor.data * tensors[name]
    #
    # total_size = 0
    # sparse_size = 0
    # for module in self.modules:
    #     for name, weight in module.named_parameters():
    #         if name in tensors:
    #             print(name, 'density:', (weight != 0).sum().item() / weight.numel())
    #             total_size += weight.numel()
    #             sparse_size += (weight != 0).sum().int().item()
    # print('Total model parameters:', total_size)
    # print(f'Total parameters under sparsity level of {sparsity}: {sparse_size / total_size}')

    return tensors


def _apply_random_weights(tensors, fix_seed=False):
    i = 0
    for key, mask in tensors.items():
        # Convert Dimension to int to avoid problems
        size = [int(x) for x in mask.shape]

        if fix_seed:
            # Fix seed for run-to-run consistency
            tensor = xavier_init(size, i)
        else:
            tensor = xavier_init(size)

        tensors[key] = tensor * mask
        i += 1

    return tensors


def erdos_renyi_random_weights_init(tensors, sparsity, erk_power_scale=1.0, fix_seed=False):
    """Erdos Renyi with Random Weights initialization.

    Args:
        - tensors: the tensors to apply sparsity on
        - sparsity: the level of sparsity [0,1)
        - erk_power_scale: ?
        - fix_seed: fix random seed

    Returns:
        - initialized Erdos Renyi sparsed tensors.
    """

    tensors = erdos_renyi_init(tensors, sparsity, erk_power_scale, fix_seed)
    return _apply_random_weights(tensors, fix_seed)


def erdos_renyi_kernel_random_weights_init(tensors, sparsity, erk_power_scale=1.0, fix_seed=False):
    """Erdos Renyi Kernel with Random Weights initialization.

    Args:
        - tensors: the tensors to apply sparsity on
        - sparsity: the level of sparsity [0,1)
        - erk_power_scale: ?
        - fix_seed: fix random seed

    Returns:
        - initialized Erdos Renyi Kernel sparsed tensors.
    """

    tensors = erdos_renyi_kernel_init(tensors, sparsity, erk_power_scale, fix_seed)
    return _apply_random_weights(tensors, fix_seed)


def snip_init(tensors, sparsity, fix_seed=False):
    return tensors


def rsensitivity_init(tensors, sparsity, fix_seed=False):
    return tensors


def binary_sampler(p, rows, cols, seed=None):
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns
      - seed: random seed

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """

    # Fix seed for run-to-run consistency
    if seed: np.random.seed(seed)

    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols, seed=None):
    """Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns
      - seed: random seed

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    """

    # Fix seed for run-to-run consistency
    if seed: np.random.seed(seed)

    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size, seed=None):
    """Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size
      - seed: random seed

    Returns:
      - batch_idx: batch index
    """

    # Fix seed for run-to-run consistency
    if seed: np.random.seed(seed)

    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx


def save_imputation_results(data_save, data_name, miss_rate, method, init, sparsity, n_nearest_features, rmse, tensors,
                            flops, folder):
    """Saves the imputed data to a csv file

    Args:
        - data_save: the data to save
        - data_name: the name of the imputed dataset
        - miss_rate: the percentage of missing datapoints
        - method: the method used (GAIN, IterativeImputer, IterativeImputerRF)
        - init: the initialization function used (GAIN only)
        - sparsity: the level of model sparsity (GAIN only)
        - n_nearest_features: the number of nearest features (Iterative Imputers only)
        - rmse: the RMSE score
        - folder: the folder to save the results in
        - tensors: the initialized tensors for the generator
    """

    # Set the folder paths and filename
    folder_inits = f'{folder}_initializations'
    folder_metrics = f'{folder}_metrics'
    folder_flops = f'{folder}_flops'
    path_metrics = f'{folder_metrics}/successes_and_failures.csv'

    if method == 'GAIN':
        filename = f'{data_name}_missrate_{miss_rate}_{method}_{init}_sparsity_{sparsity}_rmse_{rmse}'
    elif method == 'ExpectationMaximization':
        filename = f'{data_name}_missrate_{miss_rate}_{method}_rmse_{rmse}'
    else:  # Iterative Imputers
        filename = f'{data_name}_missrate_{miss_rate}_{method}_n_nearest_features_{n_nearest_features}_rmse_{rmse}'

    # Avoid overwriting if rmse is the same
    if isfile(f'{folder}/{filename}.csv'):
        i = 1
        while isfile(f'{folder}/{filename}_{i}.csv'): i += 1
        filename = f'{filename}_{i}'

    # Create the directories
    if not isdir(folder): makedirs(folder)
    if not isdir(folder_inits): makedirs(folder_inits)
    if not isdir(folder_metrics): makedirs(folder_metrics)
    if not isdir(folder_flops): makedirs(folder_flops)

    # Save the imputation
    if rmse != 'nan':
        path = f'{folder}/{filename}.csv'
        data_save = pd.DataFrame(data_save)
        data_save.to_csv(path, index=False)

        # Save the initializations
        for i in range(len(tensors)):
            tensor = tensors[i]
            path_inits = f'{folder_inits}/{filename}_G_W{i + 1}.csv'
            tensor = pd.DataFrame(tensor).mask(tensor == 0., '0').mask(tensor == 1., '1')
            tensor.to_csv(path_inits, index=False, header=False)

    # Save the metrics
    header = ['dataset', 'miss_rate', 'method', 'init', 'sparsity', 'n_nearest_features', 'successes', 'failures']
    if isfile(path_metrics):
        metrics = pd.read_csv(path_metrics)
        if not metrics.loc[
            (metrics['dataset'] == data_name)
            & (metrics['miss_rate'] == miss_rate)
            & (metrics['method'] == method)
            & (metrics['init'] == init if init else metrics['init'].isnull())
            & (metrics['sparsity'] == sparsity)
            & (metrics['n_nearest_features'] == n_nearest_features if n_nearest_features
            else metrics['n_nearest_features'].isnull())
        ].empty:
            metrics.loc[
                (metrics['dataset'] == data_name)
                & (metrics['miss_rate'] == miss_rate)
                & (metrics['method'] == method)
                & (metrics['init'] == init if init else metrics['init'].isnull())
                & (metrics['sparsity'] == sparsity)
                & (metrics['n_nearest_features'] == n_nearest_features if n_nearest_features
                   else metrics['n_nearest_features'].isnull()),
                ['successes', 'failures']
            ] += [1, 0] if rmse != 'nan' else [0, 1]
        else:
            if rmse != 'nan':
                row = pd.DataFrame([[data_name, miss_rate, method, init, sparsity, n_nearest_features, 1, 0]],
                                   columns=header)
            else:
                row = pd.DataFrame([[data_name, miss_rate, method, init, sparsity, n_nearest_features, 0, 1]],
                                   columns=header)
            metrics = pd.concat([metrics, row])
    else:
        if rmse != 'nan':
            row = [[data_name, miss_rate, method, init, sparsity, n_nearest_features, 1, 0]]
        else:
            row = [[data_name, miss_rate, method, init, sparsity, n_nearest_features, 0, 1]]
        metrics = pd.DataFrame(row, columns=header)

    # Save the success and failure count
    metrics.to_csv(path_metrics, index=False)

    # Save the flops
    flops = pd.DataFrame(flops)
    flops.to_csv(f'{folder_flops}/{filename}.csv', index=False, header=False)


def load_imputed_data(folder='imputed_data'):
    """Load the RMSE scores of the imputed data

    Args:
        - folder: the folder to find the imputed data in

    Returns:
        - a table with [data_name, miss_rate, method, initialization, sparsity, n_nearest_features, RMSE...]
    """

    concat_results = []
    if isdir(folder):
        # Get the files from the imputed data folder
        files = listdir(folder)

        # Return immediately if the folder is empty
        if len(files) == 0: return concat_results

        # Get the data_name, miss_rate, sparsity, initialization and RMSE scores from the file names
        results = []
        for f in files:
            split = f.split('_')
            data_name = split[0]
            miss_rate = float(split[2])
            method = split[3]
            if method == 'GAIN':
                init = split[4]
                sparsity = float(split[6])
                n_nearest_features = None
                rmse = float('0.' + split[8].split('.')[1])
            elif method == 'ExpectationMaximization':
                init = ''
                sparsity = 0.
                n_nearest_features = None
                rmse = float('0.' + split[5].split('.')[1])
            else:  # Iterative Imputers
                init = ''
                sparsity = 0.
                n_nearest_features = int(split[7]) if split[7] != 'None' else None
                rmse = float('0.' + split[9].split('.')[1])

            results.append([data_name, miss_rate, method, init, sparsity, n_nearest_features, rmse])

        # Concatenate the results
        res_ = None
        for result in results:
            if not res_:
                # Set res_ to result for the first item
                res_ = result
            else:
                # If the dataset, miss_rate, method, initialization and sparsity are the same, concat the RMSEs
                if res_[:6] == result[:6]:
                    res_.append(result[6])
                else:
                    concat_results.append(res_)
                    res_ = result

        # Append the final result
        concat_results.append(res_)

    return concat_results


def get_flops(theta, sparsity, init):
    """Compute the inference FLOPs of one of the models (generator or discriminator).

    Args:
        theta: the layer weights and biases of the model
        sparsity: the initial sparsity of the model
        init: the initialization used

    Returns:
        - flops: the inference FLOPs
    """
    W1, W2, W3, b1, b2, b3 = theta

    # Build the Keras layers
    W1_l = keras.layers.Dense(W1.shape[1], activation='relu')
    W1_l.build((W1.shape[1], W1.shape[0]))
    W1_l.set_weights([W1, b1])
    W1_l.kernel.name = 'G_W1'

    W2_l = keras.layers.Dense(W2.shape[1], activation='relu')
    W2_l.build((W2.shape[1], W2.shape[0]))
    W2_l.set_weights([W2, b2])
    W2_l.kernel.name = 'G_W2'

    W3_l = keras.layers.Dense(W3.shape[1], activation='sigmoid')
    W3_l.build((W3.shape[1], W3.shape[0]))
    W3_l.set_weights([W3, b3])
    W3_l.kernel.name = 'G_W3'

    layers = [W1_l, W2_l, W3_l]

    # Calculate the FLOPs
    if init in ('Dense', 'Random'):
        stats = get_stats(
            layers,
            default_sparsity=sparsity,
            method='random',
            first_layer_name='G_W1',
            last_layer_name='G_W3'
        )
    else:  # Erdos Renyi (Random Weight)
        custom_sparsities = {
            'G_W1': (W1.size - np.count_nonzero(W1)) / W1.size,
            'G_W2': (W2.size - np.count_nonzero(W2)) / W2.size,
            'G_W3': (W3.size - np.count_nonzero(W3)) / W3.size
        }

        stats = get_stats(
            layers,
            default_sparsity=sparsity,
            method='erdos_renyi',
            custom_sparsities=custom_sparsities,
            first_layer_name='G_W1',
            last_layer_name='G_W3'
        )

    flops = int(np.ceil(stats[0]))
    return flops
