"""
A bit of copy pasting from the other strategies, might refactor into an initialisation strategy and regrow strategy.
"""

from typing import Literal
import numpy as np
from strategies.strategy import Strategy
import tensorflow.compat.v1 as tf

class SnipRegrowStrategy(Strategy):
    def __init__(self, sparsity, fraction, period, vars_and_grads, sess, feed_dict, regrow : Literal["random"] | Literal["magnitude"] | None) -> None:
        self.regrow = regrow
        self.fraction = fraction
        self.sparsity = sparsity
        self.period = period
        self.masks = {}
        self.sess = sess
        self.i = 0
        self.is_first_period = True
        self.vars_and_grads = vars_and_grads
        """
        This is a dictionary from tf.Variable to symbolic gradients for the variables,
        only for the weights.
        """
        self.feed_dict = feed_dict
        """
        This is necessary for computing the SNIP scores as the TF nodes for the scores
        use the gradients of W w.r.t. the loss, which the loss needs batch data.
        """

        # Create a mask for every weight matrix
        for param in vars_and_grads.keys():
            value = sess.run(param)
            init_mask = np.where(value != 0, 1, 0).astype(np.float32)
            
            # Create a tf.Variable for the mask, trainable=False
            mask_var = tf.Variable(init_mask, dtype=tf.float32, trainable=False)
            self.masks[param] = mask_var

        self.sess.run(tf.variables_initializer(list(self.masks.values())))
        
        self.apply_masks_ops = []
        for param, mask in self.masks.items():
            self.apply_masks_ops.append(param.assign(param * mask))

        self.apply_masks_op = tf.group(*self.apply_masks_ops)

    def iteration(self, *args, **kwargs):

        # Calculate grasp on first iteration
        # Since the gradients depend on the loss, they will automatically be invoked
        if(self.i == 0):
            self.calculate_snip_mask()

        # Keep track of the period for mask transformation
        if (self.i % self.period) == 0:
            if(self.is_first_period):
                self.is_first_period = False
            elif self.regrow != None:
                if self.regrow == "random":
                    self.random_prune_and_regrow_masks()
                elif self.regrow == "magnitude":
                    self.magnitude_prune_and_regrow_masks()
                else:
                    raise Exception()
                

        # enforce the mask
        
        self.sess.run(self.apply_masks_op)

        # keep track of iteration count for mask recalculation period
        self.i += 1

    def end_train(self):
        self.sess.run(self.apply_masks_op)

    def calculate_snip_mask(self):
        """
        Recalculates the mask for the weights, using the parameters the strategy was initialised
        with. This is slow to run.
        """

        # Maps weight tf.Variable to tf operations to calculate grasp scores
        grasp_scores_dict = {}
        
        # Get parallel arrays of the weights and gradients
        weights = list(self.vars_and_grads.keys())
        grads = [self.vars_and_grads[w] for w in weights]

        # Declare the pythagoras of gradients for a subcalculation of Hg
        grad_dot = tf.add_n([tf.reduce_sum(g * g) for g in grads])

        # Compute hessian gradient product Hg as an approximation
        Hg = tf.gradients(grad_dot, weights)

        # Compute grasp scores: -g * Hg (elementwise, hadammard product)
        for w, g, h in zip(weights, grads, Hg):
            grasp_scores_dict[w] = -g * h

        # Actually calculate the grasp scores, getting a dict of weight matrix to score np.array
        grasp_scores_val = self.sess.run(grasp_scores_dict, feed_dict=self.feed_dict)

        # For every weight matrix, mask out a fraction of the weights with the lowest score
        for w, score in grasp_scores_val.items():
            mask = self.masks[w]
            threshold = np.percentile(score, self.sparsity * 100)
            new_mask = (score > threshold).astype(np.float32)
            self.sess.run(mask.assign(new_mask))

    def random_prune_and_regrow_masks(self):
        for Wtf, Mtf in self.masks.items():
            # Get current weights and masks as numpy arrays
            W = self.sess.run(Wtf)
            M = self.sess.run(Mtf)

            # Number of weights to prune/regrow
            total_ones = np.sum(M)
            k = int(self.fraction(self.i) * total_ones)

            # Prune k existing weights
            ones_idx = np.argwhere(M == 1)
            if len(ones_idx) > 0 and k > 0:
                prune_idx = ones_idx[np.random.choice(len(ones_idx), min(k, len(ones_idx)), replace=False)]
                for idx in prune_idx:
                    M[tuple(idx)] = 0

            # Regrow k new weights
            zeros_idx = np.argwhere(M == 0)
            if len(zeros_idx) > 0 and k > 0:
                regrow_idx = zeros_idx[np.random.choice(len(zeros_idx), min(k, len(zeros_idx)), replace=False)]
                for idx in regrow_idx:
                    M[tuple(idx)] = 1
                    # Xavier initialization
                    if len(W.shape) == 2:
                        fan_in, fan_out = W.shape
                    else:
                        fan_in, fan_out = np.prod(W.shape[:-1]), W.shape[-1]
                    limit = np.sqrt(6 / (fan_in + fan_out))
                    W[tuple(idx)] = np.random.uniform(-limit, limit)

            # Assign back to TensorFlow variables
            self.sess.run([Wtf.assign(W), Mtf.assign(M)])

    def magnitude_prune_and_regrow_masks(self):
        for Wtf, Mtf in self.masks.items():
            # Get current weights and masks as numpy arrays
            W = self.sess.run(Wtf)
            M = self.sess.run(Mtf)

            if(np.isnan(W).any()):
                pass

            # Number of weights to prune/regrow
            total_ones = np.sum(M)
            k = int(self.fraction(self.i) * total_ones)

            # Prune k existing weights
            ones_idx = np.argwhere(M == 1)
            if len(ones_idx) > 0 and k > 0:
                # Get absolute values of weights where mask is 1
                abs_weights = np.abs(W[tuple(ones_idx.T)])
                # Indices of the k smallest weights
                prune_local_idx = np.argsort(abs_weights)[:min(k, len(ones_idx))]
                prune_idx = ones_idx[prune_local_idx]
                for idx in prune_idx:
                    M[tuple(idx)] = 0

            # Regrow k new weights
            zeros_idx = np.argwhere(M == 0)
            if len(zeros_idx) > 0 and k > 0:
                regrow_idx = zeros_idx[np.random.choice(len(zeros_idx), min(k, len(zeros_idx)), replace=False)]
                for idx in regrow_idx:
                    M[tuple(idx)] = 1
                    # Xavier initialization
                    if len(W.shape) == 2:
                        fan_in, fan_out = W.shape
                    else:
                        fan_in, fan_out = np.prod(W.shape[:-1]), W.shape[-1]
                    limit = np.sqrt(6 / (fan_in + fan_out))
                    W[tuple(idx)] = np.random.uniform(-limit, limit)

                if(np.isnan(W).any()):
                    pass

            # Assign back to TensorFlow variables
            self.sess.run([Wtf.assign(W), Mtf.assign(M)])

