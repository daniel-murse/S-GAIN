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

"""Initializations and training strategies for S-GAIN:

This version uses TensorFlow 1.x and FP32 precision.

(1) normal_xavier_init: Normal Xavier initialization
(2) Todo: uniform_xavier_init: Uniform Xavier initialization
(3) random_init: Random initialization
(4) Todo: non-uniform random: different sparsity per layer
(5) erdos_renyi_init: Erdos Renyi initialization
(6) Todo: erdos_renyi_kernel_init: Erdos Renyi Kernel initialization
(7) get_random_weights: helper function for getting random weights
(8) erdos_renyi_random_weights_init: Erdos Renyi with Random Weights initialization
(9) Todo: erdos_renyi_kernel_random_weights_init: Erdos Renyi Kernel with Random Weights initialization
(10) Todo: snip_init: SNIP initialization
(11) Todo: grasp_init: GraSP initialization
(12) Todo: rsensitivity_init: RSensitivity initialization
"""

import numpy as np
import tensorflow as tf


def normal_xavier_init(size):
    """Normal Xavier initialization.

    :param size: vector size

    :return:
    - W: initialized normal random vector
    """

    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    W = tf.random.normal(shape=size, stddev=xavier_stddev)

    return W


def random_init(tensors, sparsity):
    """Random initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)

    :return:
    - initialized randomly pruned tensors
    """

    for i in range(len(tensors)):
        tensor = tensors[i]
        mask = np.random.choice([0, 1], size=tensor.shape, p=[sparsity, 1 - sparsity])
        tensors[i] = tensor * mask
    return tensors

def magnitude_init(tensors, sparsity):
    """Magnitude initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)

    :return:
    - initialized pruned tensors by weight magnitude
    """

    for i in range(len(tensors)):
        tensor = tensors[i]

        magnitudes = tf.abs(tensor)
        flat = tf.reshape(magnitudes, [-1])
        
        # Sort magnitudes
        sorted_mags = tf.sort(flat)
        
        # Find threshold index (weight at the sparsity percentile for its magnitude)
        # Declare the number of elements in the tensor as a tf node
        num_elements = tf.size(sorted_mags)
        # Get the index which splits the sorted weights into weights to prune or keep, by multiplying sparsity and element count
        threshold_index = tf.cast(tf.floor(sparsity * tf.cast(num_elements, tf.float32)), tf.int32)
        # Clip index in array range in case of floating point errors
        threshold_index = tf.clip_by_value(threshold_index, 0, num_elements - 1)
        
        # Get the value of the threshold
        threshold = sorted_mags[threshold_index]
        
        # Create mask (values greater than the threshold value)
        mask = tf.cast(tf.greater(magnitudes, threshold), tensor.dtype)
        
        # Apply mask
        tensors[i] = (tensor * mask)
    return tensors


def erdos_renyi_init(tensors, sparsity, erk_power_scale=1.0):
    """Erdos Renyi initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)
    :param erk_power_scale: ?

    :return:
    - initialized Erdos Renyi pruned tensors
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


def erdos_renyi_kernel_init(tensors, sparsity, erk_power_scale=1.0):
    """Erdos Renyi Kernel initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)
    :param erk_power_scale: ?

    :return:
    - tensors: initialized Erdos Renyi Kernel pruned tensors
    """

    tensors = erdos_renyi_init(tensors, sparsity, erk_power_scale)

    # Todo implement
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


def get_random_weights(tensors):
    """Helper function for getting random weights.

    :param tensors: the Erdos-Renyi (Kernel) initialized tensors to use as masks

    :return:
     - tensors: the Erdos-Renyi (Kernel) initialized tensors with random weights
    """

    i = 0
    for key, mask in tensors.items():
        # Convert Dimension to int to avoid problems
        size = [int(x) for x in mask.shape]

        tensor = normal_xavier_init(size)
        tensors[key] = tensor * mask
        i += 1

    return tensors


def erdos_renyi_random_weights_init(tensors, sparsity, erk_power_scale=1.0):
    """Erdos Renyi with Random Weights initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)
    :param erk_power_scale: ?

    :return:
    - tensors: initialized Erdos Renyi pruned tensors.
    """

    tensors = erdos_renyi_init(tensors, sparsity, erk_power_scale)
    tensors = get_random_weights(tensors)
    return tensors


def erdos_renyi_kernel_random_weights_init(tensors, sparsity, erk_power_scale=1.0):
    """Erdos Renyi Kernel with Random Weights initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)
    :param erk_power_scale: ?

    :return:
    - tensors: initialized Erdos Renyi Kernel pruned tensors.
    """

    tensors = erdos_renyi_kernel_init(tensors, sparsity, erk_power_scale)
    tensors = get_random_weights(tensors)
    return tensors
