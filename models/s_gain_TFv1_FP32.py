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
from strategies.initialisation.grasp_initialisation_strategy import GraSPInitialisationStrategy
from strategies.initialisation.initialisation_strategy import InitialisationStrategy
from strategies.initialisation.magnitude_initialisation_strategy import MagnitudeInitialisationStrategy
from strategies.initialisation.random_initialisation_strategy import RandomInitialisationStrategy
from strategies.initialisation.snip_initialisation_strategy import SNIPInitialisationStrategy
from strategies.parsing.create_count_func import create_count_func
from strategies.parsing.parse_strategies import create_dst_strategies
from strategies.pruning.magnitude_prune_strategy import MagnitudePruneStrategy
from strategies.pruning.random_prune_strategy import RandomPruneStrategy
from strategies.regrowing.random_normal_xavier_regrow_strategy import RandomNormalXavierRegrowStrategy
import tensorflow.compat.v1 as tf

from tqdm import tqdm

from monitors.monitor import Monitor
from strategies.strategy import Strategy
from utils.inits_TFv1_FP32 import magnitude_init, normal_xavier_init, random_init, erdos_renyi_init, erdos_renyi_random_weights_init
from utils.metrics import get_sparsity
from strategies.parsing.tokenise_modality import tokenise_modality
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

    # By default the generator modality should not be parsed and no strategy is created
    parse_gm = False

    # Generator variables: Data + Mask as inputs (Random noise is in missing components)
    if generator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
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

    elif generator_modality == 'RSensitivity':
        return None
    else: # NOTE HACK we use allow any modality here for simpicity in manual parsing of the name for parameters; erroneous modalities are handled later

        # Normal xavier init for the weights for all strategies; no strategy discriminates this except maybe ER but that is handled above
        G_W1 = normal_xavier_init([dim * 2, h_dim])
        G_W2 = normal_xavier_init([h_dim, h_dim])
        G_W3 = normal_xavier_init([h_dim, dim])

        # If the modality is not dense or er, it is a custom modality carrying information
        # Parse the generator modality if its not er or dense. The flag will be checked later for parsing
        # As tf variables need to be created when the strategy is parsed to count their shape
        if generator_modality not in ("dense"):
            parse_gm = True

    # else:  # This should not happen.
    #     print(f'Invalid generator modality "{generator_modality}". Exiting the program.')
    #     return None

    G_W1 = tf.Variable(G_W1)
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(G_W2)
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(G_W3)
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # By default the discriminator modality should not be parsed and no strategy is created
    parse_dm = False

    # Discriminator variables: Data + Hint as inputs
    if discriminator_modality in ('ER', 'ERK', 'ERRW', 'ERKRW'):
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
        
    elif discriminator_modality == 'RSensitivity':
        return None

    else:
        D_W1 = normal_xavier_init([dim * 2, h_dim])
        D_W2 = normal_xavier_init([h_dim, h_dim])
        D_W3 = normal_xavier_init([h_dim, dim])

        # If the modality is not dense or er, it is a custom modality carrying information
        # The flag will be checked later for parsing
        # As tf variables need to be created when the strategy is parsed to count their shape
        if(discriminator_modality not in ("dense")):
            parse_dm = True
    # else:  # This should not happen.
    #     print(f'Invalid discriminator modality "{discriminator_modality}". Exiting the program.')
    #     return None

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
    use_clipping = True
    # If we should check for clipping
    log_clipping = True and use_clipping
    # Clip epsilon, keep it small
    D_prob_clip_epsilon = 1e-8

    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss

    # Set TF nodes for clipping if using clipping
    if use_clipping:
        # Clip discriminator probabilities to avoid log(0)
        # We can either tf.clip_by_value(D_prob, D_prob_clip_epsilon, 1 - D_prob_clip_epsilon) or add D_prob_clip_epsilon in the loss functions inside the logs maybe
        # We will add if clipping is disabled to still have some safety
        D_prob_clipped = tf.clip_by_value(D_prob, D_prob_clip_epsilon, 1 - D_prob_clip_epsilon)
        # D_prob_clipped = D_prob_clipped # disables clipping the probability entirely

        # Boolean flags for logging if clipping occurred
        if log_clipping:
            D_prob_clipped_flags = tf.logical_or(D_prob < D_prob_clip_epsilon, D_prob > 1 - D_prob_clip_epsilon)

            # Percentage of clipped probabilities for feature discrimination
            D_prob_clipped_percentage = tf.reduce_mean(tf.cast(D_prob_clipped_flags, tf.float32))

        # We used to clip this, but now no more as it is not needed.
        mse_denom = tf.reduce_mean(M)
        
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / mse_denom

        # Compute GAN losses
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob_clipped) + (1 - M) * tf.log(1. - D_prob_clipped))
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob_clipped))

    # Else keep the original tf nodes with no clipping
    else:
        # add epsilon in the logs to prevent nans
        D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + D_prob_clip_epsilon) + (1 - M) * tf.log(1. - D_prob + D_prob_clip_epsilon))
        G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + D_prob_clip_epsilon))
        MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    if monitor: monitor.log_loss(np.nan, np.nan, np.nan)

    # -- S-GAIN solver ------------------------------------------------------------------------------------------------

    # We explicitly create the optimisers and solvers to capture the gradients. We need this for GraSP and SNIP

    D_optimizer = tf.train.AdamOptimizer()
    G_optimizer = tf.train.AdamOptimizer()

    D_grads_and_vars = D_optimizer.compute_gradients(D_loss, var_list=theta_D)
    G_grads_and_vars = G_optimizer.compute_gradients(G_loss, var_list=theta_G)
        
    D_solver = D_optimizer.apply_gradients(D_grads_and_vars)
    G_solver = G_optimizer.apply_gradients(G_grads_and_vars)

    # Mask variables
    
    # We need these variables for calculating masks and strategy parameters
    generator_weights = [G_W1, G_W2, G_W3]
    generator_grad_dict = {v: g for (g, v) in G_grads_and_vars}
    generator_weight_grads = [generator_grad_dict[w] for w in generator_weights]
    generator_weight_counts = [w.shape.num_elements() for w in generator_weights]
    generator_masks = [tf.Variable(tf.ones_like(w), trainable=False) for w in generator_weights]

    discriminator_weights = [D_W1, D_W2, D_W3]
    discriminator_grad_dict = {v: g for (g, v) in D_grads_and_vars}
    discriminator_weight_grads = [discriminator_grad_dict[w] for w in discriminator_weights]
    discriminator_weight_counts = [w.shape.num_elements() for w in discriminator_weights]
    discriminator_masks = [tf.Variable(tf.ones_like(w), trainable=False) for w in discriminator_weights]

    # -- S-GAIN training ----------------------------------------------------------------------------------------------

    if verbose: print('Training S-GAIN...')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Initialise the DST strategy

    # Generator DST 

    # Parse the modality if it carries information describing a DST (a custom modality)
    if parse_gm:
        generator_init_strategy, generator_prune_strategy, generator_regrow_strategy = \
            create_dst_strategies(generator_modality, generator_sparsity, generator_weight_counts, iterations)
    # Else use no strategy (dense or er was used as the modality)
    else:
        generator_init_strategy = None
        generator_prune_strategy = None
        generator_regrow_strategy = None

    # This op will apply the masks to the weights. We need to run this after every training iteration to stop updates to the masked params
    # We can cache this op as long as our tf.Variable instances remain the same (which they do)
    generator_apply_masks_op = tf.group(*[
        tf.assign(w, w * m)
        for w, m in zip(generator_weights, generator_masks)
    ])
    
    # Init the mask if the init strategy exits
    if generator_init_strategy is not None:
        # Optionally use a mini batch if the strategy requires it (snip, grasp)
        if(generator_init_strategy.get_requires_mini_batch()):
            mask_updates = generator_init_strategy.get_tf_mask_initialisation_tensors(generator_weights, generator_weight_grads)
            assign_ops = [tf.assign(mask_var, new_mask) for mask_var, new_mask in zip(generator_masks, mask_updates)]
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
            sess.run(assign_ops, feed_dict={M: M_mb, X: X_mb, H: H_mb})
        # Else just get the mask from the weights (random/magnitude prune)
        else:
            mask_updates = generator_init_strategy.get_tf_mask_initialisation_tensors(generator_weights)
            assign_ops = [tf.assign(mask_var, new_mask) for mask_var, new_mask in zip(generator_masks, mask_updates)]
            sess.run(assign_ops)

    # Discriminator DST
    
    # Parse the modality if required by the modality
    if parse_dm:
        discriminator_init_strategy, discriminator_prune_strategy, discriminator_regrow_strategy = \
            create_dst_strategies(discriminator_modality, discriminator_sparsity, discriminator_weight_counts, iterations)
    # Else use no strategy (dense or er was used as the modality)
    else:
        discriminator_init_strategy = None
        discriminator_prune_strategy = None
        discriminator_regrow_strategy = None

    # This op will apply the masks to the weights. We need to run this after every training iteration to stop updates to the masked params
    # We can cache this op as long as our tf.Variable instances remain the same (which they do)
    discriminator_apply_masks_op = tf.group(*[
        tf.assign(w, w * m)
        for w, m in zip(discriminator_weights, discriminator_masks)
    ])
    
    # Init the mask if the init strategy exits
    if discriminator_init_strategy is not None:
        # Optionally use a mini batch if the strategy requires it (snip, grasp)
        if(discriminator_init_strategy.get_requires_mini_batch()):
            mask_updates = discriminator_init_strategy.get_tf_mask_initialisation_tensors(discriminator_weights, discriminator_weight_grads)
            assign_ops = [tf.assign(mask_var, new_mask) for mask_var, new_mask in zip(discriminator_masks, mask_updates)]
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
            sess.run(assign_ops, feed_dict={M: M_mb, X: X_mb, H: H_mb})
        # Else just get the mask from the weights (random/magnitude prune)
        else:
            mask_updates = discriminator_init_strategy.get_tf_mask_initialisation_tensors(discriminator_weights)
            assign_ops = [tf.assign(mask_var, new_mask) for mask_var, new_mask in zip(discriminator_masks, mask_updates)]
            sess.run(assign_ops)

    # Training loop

    # Apply the mask to start
    sess.run(discriminator_apply_masks_op)

    # The generator mask was already applied above (check if uncertain)

    for it in tqdm(range(iterations)):

        # Generator DST enforcement
        # Prune if the prune strategy exists
        if generator_prune_strategy is not None:
            # Get the updates to the pruned masks for this iteration
            pruned_masks = generator_prune_strategy.get_tf_pruned_mask_tensors(it, generator_weights, generator_masks)
            # If there are updates, run them
            if pruned_masks is not None:
                generator_prune_ops = []
                generator_prune_ops += [
                    tf.assign(mask_var, new_mask)
                    for mask_var, new_mask in zip(generator_masks, pruned_masks)
                ]
                sess.run(tf.group(*generator_prune_ops))

        # Same for the regrow, but both masks and weights can be changed
        if generator_regrow_strategy is not None:
            regrow_result = generator_regrow_strategy.get_tf_regrowed_mask_and_weight_tensors(
                it, generator_weights, generator_masks
            )
            if regrow_result is not None:
                print("regrowing")
                new_weights, new_masks = zip(*regrow_result) if regrow_result else ([], [])

                generator_regrow_ops = []

                if new_masks is not None:
                    generator_regrow_ops += [
                        tf.assign(mask_var, new_mask)
                        for mask_var, new_mask in zip(generator_masks, new_masks)
                    ]

                if new_weights is not None:
                    generator_regrow_ops += [
                        tf.assign(weight_var, new_weight)
                        for weight_var, new_weight in zip(generator_weights, new_weights)
                    ]

                sess.run(generator_regrow_ops)

        # Enforce the mask before calculating sparsity and the forward and backward pass
        sess.run(generator_apply_masks_op)

        # Do the same for the discriminator

        # Prune if the prune strategy exists
        if discriminator_prune_strategy is not None:
            # Get the updates to the pruned masks for this iteration
            pruned_masks = discriminator_prune_strategy.get_tf_pruned_mask_tensors(it, discriminator_weights, discriminator_masks)
            # If there are updates, run them
            if pruned_masks is not None:
                discriminator_prune_ops = []
                discriminator_prune_ops += [
                    tf.assign(mask_var, new_mask)
                    for mask_var, new_mask in zip(discriminator_masks, pruned_masks)
                ]
                sess.run(tf.group(*discriminator_prune_ops))

        # Same for the regrow, but both masks and weights can be changed
        if discriminator_regrow_strategy is not None:
            regrow_result = discriminator_regrow_strategy.get_tf_regrowed_mask_and_weight_tensors(
                it, discriminator_weights, discriminator_masks
            )
            if regrow_result is not None:
                print("regrowing")
                new_weights, new_masks = zip(*regrow_result) if regrow_result else ([], [])

                discriminator_regrow_ops = []

                if new_masks is not None:
                    discriminator_regrow_ops += [
                        tf.assign(mask_var, new_mask)
                        for mask_var, new_mask in zip(discriminator_masks, new_masks)
                    ]

                if new_weights is not None:
                    discriminator_regrow_ops += [
                        tf.assign(weight_var, new_weight)
                        for weight_var, new_weight in zip(discriminator_weights, new_weights)
                    ]

                sess.run(discriminator_regrow_ops)

        # Enforce the mask before calculating sparsity and the forward and backward pass
        sess.run(discriminator_apply_masks_op)
        
        # Log sparsity
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
        fetches = [D_solver, D_loss_temp]
        if(use_clipping and log_clipping and monitor is not None): fetches.extend([D_prob_clipped_percentage])

        _, D_loss_curr, *D_optional = sess.run(fetches, feed_dict={M: M_mb, X: X_mb, H: H_mb})
        

        fetches = [G_solver, G_loss_temp, MSE_loss]
        if(use_clipping and log_clipping and monitor is not None): fetches.extend([D_prob_clipped_percentage])

        _, G_loss_curr, MSE_loss_curr, *G_optional = sess.run(fetches,
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
        
        if use_clipping and log_clipping:

            if monitor is not None:
                monitor.log_clip(G_optional[0], D_optional[0])

         # Detect NANs if set
        if detect_nans:
            has_nans = False
            if (np.isnan(D_loss_curr) or np.isnan(G_loss_curr) or np.isnan(MSE_loss_curr)):
                has_nans = True
                print(f"[NaN DETECTED] losses at iter {it}: "
                    f"D_loss={D_loss_curr}, G_loss={G_loss_curr}, MSE={MSE_loss_curr}")

            if has_nans:
                # NOTE currently nans are detected by a nan rmse (afawk)
                print("Breaking...")
                break

        if monitor: monitor.log_loss(G_loss_curr, D_loss_curr, MSE_loss_curr)

    # Reinforce the mask for the last sparsity calculation after the last training forward and backward pass
    sess.run(generator_apply_masks_op)
    sess.run(discriminator_apply_masks_op)

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
