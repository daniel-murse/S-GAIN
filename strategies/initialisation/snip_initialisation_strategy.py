"""
Mask initialisation strategy that for SNIP.
"""

import numpy as np
import tensorflow as tf
from strategies.initialisation.initialisation_strategy import InitialisationStrategy

class SNIPInitialisationStrategy(InitialisationStrategy):
    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def get_tf_mask_initialisation_tensors(self, weight_tensors, gradient_tensors):
        return tf_create_snip_masks(self.sparsity, weight_tensors, gradient_tensors)
    
    def get_requires_mini_batch(self):
        return True
    
def tf_create_snip_scores(weights, grads):
    snip_scores = []
    
    # |gj (w; D)| 
    # the snip score numerator
    # The derrivative of the loss w.r.t connection strength using minibatch D
    # mathematically simplified to: wj * gj
    for w, g in zip(weights, grads):
        snip_scores.append(tf.abs(g*w))

    
    # The SNIP normalization constant
    # sum for all k |gk(w;Db)|
    snip_norm = tf.add_n(tf.reduce_sum(s) for s in snip_scores)
    
    # apply the normalization
    for i, (w, g) in enumerate(zip(weights, grads)):
        snip_scores[i] = snip_scores[i] / snip_norm
    
    return snip_scores
    
def tf_create_snip_masks(sparsity, weights, grads):
    
    snip_scores = tf_create_snip_scores(weights, grads)
    
    new_masks = []
    for i, w in enumerate(weights):
        score = snip_scores[i]

        # Compute percentile threshold in tf
        flat_scores = tf.reshape(score, [-1])
        n = tf.size(flat_scores)
        threshold_index = tf.cast(tf.round(tf.cast(n, tf.float32) * (1-sparsity)), tf.int32)
        threshold_index = tf.clip_by_value(threshold_index, 1, n - 1)

        sorted_scores = tf.sort(flat_scores)
        threshold = sorted_scores[-threshold_index]

        # Create new binary mask tensor
        new_mask = tf.cast(score > threshold, tf.float32)

        new_masks.append(new_mask)

    # Return the list of mask tensors
    return new_masks