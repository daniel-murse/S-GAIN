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

"""S-GAIN.

This version uses TensorFlow 1.x and FP32 precision.

We adapted the original GAIN code for our work:
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets",
           ICML, 2018.
Paper Link: https://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
"""

import numpy as np
from strategies.snip_strategy import SnipStrategy
import tensorflow.compat.v1 as tf

from tqdm import tqdm

from monitors.monitor import Monitor
from strategies.grasp_strategy import GraspStrategy
from strategies.magnitude_strategy import MagnitudeStrategy
from strategies.random_strategy import RandomStrategy
from strategies.strategy import Strategy
from utils.inits_TFv1_FP32 import normal_xavier_init, random_init, erdos_renyi_init, erdos_renyi_random_weights_init
from utils.metrics import get_sparsity
from utils.utils import binary_sampler, uniform_sampler, sample_batch_index, normalization, renormalization, rounding

tf.disable_v2_behavior()


def s_gain(miss_data_x, batch_size=128, hint_rate=0.9, alpha=100, iterations=10000,
           generator_sparsity=0, generator_modality='dense', discriminator_sparsity=0, discriminator_modality='dense',
           verbose=False, no_model=True, monitor=None):
    """Impute the missing values in miss_data_x.

    :param miss_data_x: the data with missing values
    :param batch_size: the number of samples in mini-batch
    :param hint_rate: the hint probability
    :param alpha: the hyperparameter
    :param iterations: the number of training iterations (epochs)
    :param generator_sparsity: the probability of sparsity in the generator
    :param generator_modality: the initialization and pruning and regrowth strategy of the generator
    :param discriminator_sparsity: the probability of sparsity in the discriminator
    :param discriminator_modality: the initialization and pruning and regrowth strategy of the discriminator
    :param verbose: enable verbose output to console
    :param no_model: don't save the trained model
    :param monitor: the monitor object used for the measurements

    :return:
    - imputed_data_x: the imputed data
    """

    if verbose: print('Starting GAIN...')

    # Start monitors
    if monitor is not None:
        monitor.init_monitor()
        monitor.start_all_monitors()

    # Reshape the missing data
    shape = miss_data_x.shape
    reshaped = False
    if len(shape) > 2:
        if verbose: print('Reshaping missing data matrix to 2D...')
        new_shape = shape[0], np.prod(shape[1:])
        miss_data_x = np.reshape(miss_data_x, new_shape)
        reshaped = True

    # Define the mask matrix
    data_mask = 1 - np.isnan(miss_data_x)

    # Parameters
    no, dim = miss_data_x.shape
    h_dim = int(dim)

    # Normalization
    if verbose: print('Normalizing data...')
    norm_data_x, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data_x, False)

    # -- S-GAIN architecture ------------------------------------------------------------------------------------------

    # Input placeholders
    X = tf.placeholder(tf.float32, shape=[None, dim])  # Data vector
    M = tf.placeholder(tf.float32, shape=[None, dim])  # Mask vector
    H = tf.placeholder(tf.float32, shape=[None, dim])  # Hint vector

    # Generator variables: Data + Mask as inputs (Random noise is in missing components)
    if generator_modality in ('dense', 'random', 'GraSP'):
        G_W1 = normal_xavier_init([dim * 2, h_dim])
        G_W2 = normal_xavier_init([h_dim, h_dim])
        G_W3 = normal_xavier_init([h_dim, dim])

        if generator_modality == 'random':
            G_W1, G_W2, G_W3 = random_init([G_W1, G_W2, G_W3], generator_sparsity)

    elif generator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
        G_Ws = {
            'G_W1': np.zeros([dim * 2, h_dim]),
            'G_W2': np.zeros([h_dim, h_dim]),
            'G_W3': np.zeros([h_dim, dim])
        }

        if generator_modality == 'ER':
            G_W1, G_W2, G_W3 = erdos_renyi_init(G_Ws, generator_sparsity).values()
        elif generator_modality == 'ERK':
            return None
        elif generator_modality == 'ERRW':
            G_W1, G_W2, G_W3 = erdos_renyi_random_weights_init(G_Ws, generator_sparsity).values()
        else:  # ERKRW
            return None

    elif generator_modality == 'SNIP':
        return None

    elif generator_modality == 'GraSP':
        print("Using GraSP") # Do not return None

    elif generator_modality == 'RSensitivity':
        return None

    else:  # This should not happen.
        print(f'Invalid generator modality "{generator_modality}". Exiting the program.')
        return None

    G_W1 = tf.Variable(G_W1)
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(G_W2)
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(G_W3)
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Discriminator variables: Data + Hint as inputs
    if discriminator_modality in ('dense', 'random', 'GraSP'):
        D_W1 = normal_xavier_init([dim * 2, h_dim])
        D_W2 = normal_xavier_init([h_dim, h_dim])
        D_W3 = normal_xavier_init([h_dim, dim])

        if discriminator_modality == 'random':
            D_W1, D_W2, D_W3 = random_init([D_W1, D_W2, D_W3], discriminator_sparsity)

    elif discriminator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
        D_Ws = {
            'D_W1': np.zeros([dim * 2, h_dim]),
            'D_W2': np.zeros([h_dim, h_dim]),
            'D_W3': np.zeros([h_dim, dim])
        }

        if discriminator_modality == 'ER':
            D_W1, D_W2, D_W3 = erdos_renyi_init(D_Ws, discriminator_sparsity).values()
        elif discriminator_modality == 'ERK':
            return None
        elif discriminator_modality == 'ERRW':
            D_W1, D_W2, D_W3 = erdos_renyi_random_weights_init(D_Ws, discriminator_sparsity).values()
        else:  # ERKRW
            return None

    elif discriminator_modality == 'SNIP':
        return None

    elif discriminator_modality == 'GraSP':
        return None

    elif discriminator_modality == 'RSensitivity':
        return None

    else:  # This should not happen.
        print(f'Invalid discriminator modality "{discriminator_modality}". Exiting the program.')
        return None

    D_W1 = tf.Variable(D_W1)
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(D_W2)
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(D_W3)
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # -- S-GAIN functions ---------------------------------------------------------------------------------------------

    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.add(tf.matmul(inputs, G_W1), G_b1))
        G_h2 = tf.nn.relu(tf.add(tf.matmul(G_h1, G_W2), G_b2))
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.add(tf.matmul(G_h2, G_W3), G_b3))
        return G_prob

    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        # MinMax normalized output
        D_prob = tf.nn.sigmoid(tf.matmul(D_h2, D_W3) + D_b3)
        return D_prob

    # -- S-GAIN structure ---------------------------------------------------------------------------------------------

    # Flags for adjusting the training loop

    # Should NANs in tensorflow be detected?
    detect_nans = True
    # Should we clip the NN outputs and losses to avoid NANs?
    use_clipping = False
    # Clip scale threshold hyperparameter for gradient clipping
    clip_norm = 1.0
    log_clipping = True and use_clipping
    # todo add epsilon constants for clipping

    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss

    # Set TF nodes for clipping
    if use_clipping:
        # Clip discriminator probabilities to avoid log(0)
        D_prob_clipped = tf.clip_by_value(D_prob, 1e-7, 1 - 1e-7)
        # D_prob_clipped = D_prob_clipped # disables clipping the probability entirely
        # Boolean flags for logging if clipping occurred
        if log_clipping:
            D_prob_clipped_flags = tf.logical_or(D_prob < 1e-7, D_prob > 1 - 1e-7)

            # Percentage of clipped probabilities for feature discrimination
            D_prob_clipped_percentage = tf.reduce_mean(tf.cast(D_prob_clipped_flags, tf.float32))

        # Compute MSE loss without division by zero. But, we are sure this is not an issue
        # And this denominator is never 0 even without the epsilon
        mse_denom = tf.reduce_mean(M) + 1e-8
        if log_clipping:
            mse_clipped_flag = tf.less(tf.reduce_mean(M), 1e-8)

            # We will treat this as a percentage for logging with the monitor
            mse_clipped_percentage = tf.cast(mse_clipped_flag, tf.float32)
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / mse_denom
        # TODO ADD INSTEAD OF CLIP
        # Compute GAN losses
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob_clipped) + (1 - M) * tf.log(1. - D_prob_clipped))
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob_clipped))

    # Else keep the original tf nodes
    else:
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    if monitor: monitor.log_loss(np.nan, np.nan, np.nan)

    # -- S-GAIN solver ------------------------------------------------------------------------------------------------

    # We explicitly create the optimisers and solvers to capture the gradients for checking NANs

    D_optimizer = tf.train.AdamOptimizer()
    G_optimizer = tf.train.AdamOptimizer()

    D_grads_and_vars = D_optimizer.compute_gradients(D_loss, var_list=theta_D)
    G_grads_and_vars = G_optimizer.compute_gradients(G_loss, var_list=theta_G)

    # Clip the gradients and vars if using clipping
    if use_clipping:
        # Compute norms before clipping so we can check if clipping occurred
        D_grad_norms = [tf.norm(g) for g, v in D_grads_and_vars if g is not None]
        G_grad_norms = [tf.norm(g) for g, v in G_grads_and_vars if g is not None]

        # Boolean flags for whether clipping happened
        if log_clipping:
            # Did any weight matrix get clipped; tihs is not how many parameters (weights) get clipped
            # its how many variables (weight matrices) contain parameters that got clipped
            # its why its constant in the graph
            # todo arghh we could calculate this as a % of weights per variable 
            D_clipped_flags = [tf.greater(n, clip_norm) for n in D_grad_norms]
            G_clipped_flags = [tf.greater(n, clip_norm) for n in G_grad_norms]

            # TF operations to calculate the percentage of gradients clipped
            D_clipped_percentage = tf.reduce_mean(tf.cast(D_clipped_flags, tf.float32))
            G_clipped_percentage = tf.reduce_mean(tf.cast(G_clipped_flags, tf.float32))


        # Apply clipping
        D_grads_and_vars = [(tf.clip_by_norm(g, clip_norm), v) for g, v in D_grads_and_vars if g is not None]
        G_grads_and_vars = [(tf.clip_by_norm(g, clip_norm), v) for g, v in G_grads_and_vars if g is not None]
        
    D_solver = D_optimizer.apply_gradients(D_grads_and_vars)
    G_solver = G_optimizer.apply_gradients(G_grads_and_vars)

    # -- S-GAIN training ----------------------------------------------------------------------------------------------

    if verbose: print('Training S-GAIN...')

    # Grafted random pruning, no regrow
    # "alpha" taken as a paremeter ("the hyperparameter") might be used for this
    prune_period = 100


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Initialise the DST strategy

    generator_strategy : Strategy | None = None

    if generator_modality == "random":

        generator_params = [G_W1, G_W2, G_W3]

        generator_strategy = RandomStrategy(0.2, prune_period, generator_params, sess)

    elif generator_modality == "GraSP":

        generator_params = [G_W1, G_W2, G_W3]

        # Map G weights to G weight gradients. Remember they are symbolic tensors
        G_W_vars_and_grads = { v : g for g, v in G_grads_and_vars if v in generator_params }

        # Sample batch
        # NOTE This seems to sample batches randomly instead of sequentially using shuffled data
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_mask[batch_idx, :]

        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        feed_dict = {M: M_mb, X: X_mb, H: H_mb}

        # Create the grasp strategy. Note: whether to periodically recompute the grasp mask,
        # and if so, whether to use a differend feed_dict (batch) for grasp scores or the same one
        generator_strategy = SnipStrategy(generator_sparsity, prune_period, G_W_vars_and_grads, sess, feed_dict)

    for it in tqdm(range(iterations)):

        if generator_strategy is not None:
            generator_strategy.iteration()
                    
        if monitor:
            monitor.log_imputation_time()

            G_sparsities = get_sparsity(sess.run(theta_G))
            D_sparsities = get_sparsity(sess.run(theta_D))
            monitor.log_sparsity(G_sparsities, D_sparsities)

        # Sample batch
        # NOTE This seems to sample batches randomly instead of sequentially using shuffled data
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_mask[batch_idx, :]

        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

        # Sample hint vectors
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        # As we run D and G, we also may get clipping data for logging
        fetches = [D_solver, D_loss_temp, [g for g, v in D_grads_and_vars]]
        if(use_clipping and log_clipping and monitor is not None): fetches.extend([D_clipped_percentage, D_prob_clipped_percentage, mse_clipped_percentage])

        _, D_loss_curr, D_grads, *D_optional = sess.run(fetches, feed_dict={M: M_mb, X: X_mb, H: H_mb})
        

        fetches = [G_solver, G_loss_temp, MSE_loss, [g for g in G_grads_and_vars]]
        if(use_clipping and log_clipping and monitor is not None): fetches.extend([G_clipped_percentage, D_prob_clipped_percentage, mse_clipped_percentage])

        _, G_loss_curr, MSE_loss_curr, G_grads, *G_optional = sess.run(fetches,
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
        
        if use_clipping and log_clipping:
            D_clipping = D_optional[0]
            G_clipping = G_optional[0]

            if monitor is not None:
                monitor.log_clip(G_optional[0], D_optional[0], G_optional[2], D_optional[2], G_optional[1], D_optional[1])

            # # Check and print indices where clipping occurred
            # clipped_d_indices = [i for i, val in enumerate(D_optional) if val > 0]
            # clipped_g_indices = [i for i, val in enumerate(G_optional) if val > 0]

            # if clipped_d_indices or clipped_g_indices:
            #     print(f"[Clipping] occurred at iter {it}")
            #     if clipped_d_indices:
            #         print(f"  D_optional clipped at indices: {clipped_d_indices}")
            #     if clipped_g_indices:
            #         print(f"  G_optional clipped at indices: {clipped_g_indices}")

         # Detect NANs if set
        if detect_nans:
            has_nans = False
            if (np.isnan(D_loss_curr) or np.isnan(G_loss_curr) or np.isnan(MSE_loss_curr)):
                has_nans = True
                print(f"[NaN DETECTED] losses at iter {it}: "
                    f"D_loss={D_loss_curr}, G_loss={G_loss_curr}, MSE={MSE_loss_curr}")

            for g in D_grads + G_grads:
                if np.isnan(g).any():
                    has_nans = True
                    print(f"[NaN DETECTED] gradient at iter {it}")
                    break

            for param in sess.run(theta_D + theta_G):
                if np.isnan(param).any():
                    has_nans = True
                    print(f"[NaN DETECTED] weights at iter {it}")
                    break
            
            if has_nans:
                break

        if monitor: monitor.log_loss(G_loss_curr, D_loss_curr, MSE_loss_curr)

    if(generator_strategy is not None):  
        generator_strategy.end_train()

    if monitor:
        monitor.log_imputation_time()

        G_sparsities = get_sparsity(sess.run(theta_G))
        D_sparsities = get_sparsity(sess.run(theta_D))
        monitor.log_sparsity(G_sparsities, D_sparsities)

    if verbose: print('Finished training.')

    # Return the imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_mask
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data_x = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    imputed_data_x = data_mask * norm_data_x + (1 - data_mask) * imputed_data_x

    # -----------------------------------------------------------------------------------------------------------------

    # Renormalization
    if verbose: print('Re-normalizing data...')
    imputed_data_x = renormalization(imputed_data_x, norm_parameters)

    # Rounding
    if verbose: print('Rounding data...')
    imputed_data_x = rounding(imputed_data_x, miss_data_x)

    # Reshaping
    if reshaped:
        if verbose: print('Reshaping the imputed data to the original shape...')
        imputed_data_x = np.reshape(imputed_data_x, shape)

    # Stop monitor
    if monitor:
        monitor.log_imputation_time()
        monitor.log_rmse(imputed_data_x)
        monitor.stop_all_monitors()

        # Save model
        if not no_model: monitor.set_model(sess.run(theta_G), sess.run(theta_D))

    if verbose: print('Stopped S-GAIN.')

    return imputed_data_x
