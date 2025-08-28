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

Todo: This version uses TensorFlow 2.x and INT8 precision.

(1) normal_xavier_init: Normal Xavier initialization
(2) Todo: uniform_xavier_init: Uniform Xavier initialization
(3) random_init: Random initialization
(4) Todo: non-uniform random: different sparsity per layer
(5) Todo: erdos_renyi_init: Erdos Renyi initialization
(6) Todo: erdos_renyi_kernel_init: Erdos Renyi Kernel initialization
(7) Todo: get_random_weights: helper function for getting random weights
(8) Todo: erdos_renyi_random_weights_init: Erdos Renyi with Random Weights initialization
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
    W = tf.random.normal(shape=size, stddev=xavier_stddev)  # Todo: dtype (q)int8

    return W


# def normal_xavier_init(W1_shape, W2_shape, W3_shape):
#     """Normal Xavier initialization.
#
#     :param W1_shape: the shape of the first kernel
#     :param W2_shape: the shape of the second kernel
#     :param W3_shape: the shape of the third kernel
#
#     :return:
#     - W1: the first normal xavier initialized kernel
#     - W2: the second normal xavier initialized kernel
#     - W3: the third normal xavier initialized kernel
#     """
#
#     # Input dimensions
#     W1_size = W1_shape[0]
#     W2_size = W2_shape[0]
#     W3_size = W3_shape[0]
#
#     # Xavier standard deviations
#     W1_xavier_stddev = tf.sqrt(2. / W1_size)
#     W2_xavier_stddev = tf.sqrt(2. / W2_size)
#     W3_xavier_stddev = tf.sqrt(2. / W3_size)
#
#     # Random normal initialized kernels
#     W1 = tf.random_normal_initializer(shape=W1_shape, stddev=W1_xavier_stddev)
#     W2 = tf.random_normal_initializer(shape=W2_shape, stddev=W2_xavier_stddev)
#     W3 = tf.random_normal_initializer(shape=W3_shape, stddev=W3_xavier_stddev)
#
#     return W1, W2, W3


def random_init(tensors, sparsity):
    """Random initialization.

    :param tensors: the tensors to apply sparsity on
    :param sparsity: the level of sparsity [0,1)

    :return:
    - tensors: initialized randomly pruned tensors
    """

    for i in range(len(tensors)):
        tensor = tensors[i]
        mask = np.random.choice([0, 1], size=tensor.shape, p=[sparsity, 1 - sparsity])
        tensors[i] = tensor * mask
    return tensors
