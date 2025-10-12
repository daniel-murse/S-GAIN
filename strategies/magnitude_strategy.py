import numpy as np
import tensorflow.compat.v1 as tf
from strategies.strategy import Strategy
from utils.inits_TFv1_FP32 import normal_xavier_init

class MagnitudeStrategy(Strategy):
    def __init__(self, fraction, period, generator_params, sess) -> None:
        super().__init__()
        self.fraction = fraction
        self.period = period
        self.masks = {}
        self.sess = sess
        self.i = 0
        self.is_first_period = True

        # Create a mask for every weight matrix
        for param in generator_params:
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
        # Keep track of the period for mask transformation
        if (self.i % self.period) == 0:
            if(self.is_first_period):
                self.is_first_period = False
            else:
                self.prune_and_regrow_masks()

        # enforce the mask
        
        self.sess.run(self.apply_masks_op)

        # keep track of iteration count for mask recalculation period
        self.i += 1

    def end_train(self):
        self.sess.run(self.apply_masks_op)

    def prune_and_regrow_masks(self):
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

