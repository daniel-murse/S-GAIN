"""
A DST strategy that uses independent generator and discriminator strategies, splitting strategies into 3 components:
 - initialisation of the mask
 - pruning of the mask
 - regrowing of the mask
"""

import tensorflow as tf
from strategies.parsing.parse_strategies import create_dst_strategies
from strategies.strategy import Strategy


class SimpleStrategy(Strategy):

    def __init__(self, sess, weights, gradients, masks, modality, sparsity, iterations):
        weight_counts = [w.shape.num_elements() for w in weights]
        self.init_strategy, self.prune_strategy, self.regrow_strategy = \
            create_dst_strategies(modality, sparsity, weight_counts, iterations)
        self.weights = weights
        self.masks = masks
        self.gradients = gradients
        self.apply_masks_op = tf.group(*[
        tf.assign(w, w * m)
        for w, m in zip(weights, masks)
    ])

    def start_train(self, *args, **kwargs):
        return super().start_train(*args, **kwargs)
    
    def iteration(self, *args, **kwargs):
        return super().iteration(*args, **kwargs)
    
    def end_train(self, *args, **kwargs):
        return super().end_train(*args, **kwargs)

def create_simple_strategy():
    pass