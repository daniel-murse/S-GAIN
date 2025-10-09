import numpy as np
from strategies.strategy import Strategy
import tensorflow.compat.v1 as tf

class SnipStrategy(Strategy):
    def __init__(self, fraction, period, vars_and_grads, sess, feed_dict) -> None:
        self.fraction = fraction
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
        This is necessary for computing the GraSP scores as the TF nodes for the scores
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

        # Calculate snip on first iteration
        if(self.i == 0):
            self.calculate_snip_mask()

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

    def calculate_snip_mask(self):
        """
        Recalculates the mask for the weights, using the parameters the strategy was initialised
        with. This is slow to run.
        """

        # Maps weight tf.Variable to tf operations to calculate grasp scores
        snip_scores_dict = {}

        
        # Get parallel arrays of the weights and gradients
        weights = list(self.vars_and_grads.keys())
        grads = [self.vars_and_grads[w] for w in weights]
        
      
        # |gj (w; D)| 
        # the snip score numerator
        # The derrivative of the loss w.r.t connection strength using minibatch D
        # mathematically simplified to: wj * gj
        for w, g in zip(weights, grads):
            snip_scores_dict[w] = tf.abs(g*w)

        
        # The SNIP normalization constant
        # sum for all k |gk(w;Db)|
        snip_norm = tf.add_n(tf.reduce_sum(s) for s in snip_scores_dict.values())
        
        # apply the normalization
        for w, g in zip(weights, grads):
            snip_scores_dict[w] = snip_scores_dict[w] / snip_norm
        
        # Calculate SNIP score, getting a dict of weight matrix to score np.array
        snip_scores_val = self.sess.run(snip_scores_dict, feed_dict=self.feed_dict)

        # For every weight matrix, mask out a fraction of the weights with the lowest score
        for w, score in snip_scores_val.items():
            mask = self.masks[w]
            threshold = np.percentile(score, self.fraction * 100)
            new_mask = (score > threshold).astype(np.float32)
            self.sess.run(mask.assign(new_mask))


    def prune_and_regrow_masks(self):
        for Wtf, Mtf in self.masks.items():
            # Get current weights and masks as numpy arrays
            W = self.sess.run(Wtf)
            M = self.sess.run(Mtf)

            # Nothing here now

            # Assign back to TensorFlow variables
            self.sess.run([Wtf.assign(W), Mtf.assign(M)])
