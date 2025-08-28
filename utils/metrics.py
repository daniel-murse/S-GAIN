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

"""Metrics calculations for S-GAIN:

(1) get_rmse: evaluate the imputed data in terms of RMSE
(2) get_sparsity: compute the sparsity of the model
(3) get_flops: compute the inference FLOPs
"""

import keras

import numpy as np

from utils.utils import normalization
from utils.flops.sparse_utils import get_stats


def get_rmse(data_x, imputed_data_x, data_mask, round=False):
    """Compute the RMSE between the original data and the imputed data.

    :param data_x: the original data (without missing values)
    :param imputed_data_x: the imputed data
    :param data_mask: the indicator matrix for missing elements
    :param round: whether to round or not

    :return: the Root Mean Squared Error (rounded to 4 decimals)
    """

    data_x, norm_parameters = normalization(data_x)
    imputed_data, _ = normalization(imputed_data_x, norm_parameters)

    nominator = np.sum(((1 - data_mask) * data_x - (1 - data_mask) * imputed_data) ** 2)
    denominator = np.sum(1 - data_mask)
    RMSE = np.sqrt(nominator / float(denominator))
    if round: RMSE = f'{RMSE:.4f}'

    return RMSE


def get_sparsity(theta):
    """Compute the actual sparsity of one of the models (generator or discriminator).

    :param theta: the layer weights and biases of the model

    :return:
    - M_sparsity: the total sparsity of the model
    - W1_sparsity: the sparsity of the first layer of the model
    - W2_sparsity: the sparsity of the second layer of the model
    - W3_sparsity: the sparsity of the third layer of the model
    """

    W1, W2, W3, b1, b2, b3 = theta

    W1_size = np.size(W1)
    W1_nzc = np.count_nonzero(W1)
    W1_sparsity = (W1_size - W1_nzc) / W1_size

    W2_size = np.size(W2)
    W2_nzc = np.count_nonzero(W2)
    W2_sparsity = (W2_size - W2_nzc) / W2_size

    W3_size = np.size(W3)
    W3_nzc = np.count_nonzero(W3)
    W3_sparsity = (W3_size - W3_nzc) / W3_size

    M_size = W1_size + W2_size + W3_size
    M_nzc = W1_nzc + W2_nzc + W3_nzc
    M_sparsity = (M_size - M_nzc) / M_size

    return M_sparsity, W1_sparsity, W2_sparsity, W3_sparsity


def get_flops(theta, sparsity, init):
    """Compute the inference FLOPs of one of the models (generator or discriminator).

    :param theta: the layer weights and biases of the model
    :param sparsity: the initial sparsity of the model
    :param init: the initialization used

    :return:
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