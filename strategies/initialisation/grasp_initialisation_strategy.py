"""
Mask initialisation strategy that for GraSP.
"""

import numpy as np
import tensorflow as tf
from strategies.initialisation.initialisation_strategy import InitialisationStrategy

class GraSPInitialisationStrategy(InitialisationStrategy):
    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def get_tf_mask_initialisation_tensors(self, weight_tensors, gradient_tensors):
        return tf_create_grasp_masks(self.sparsity, weight_tensors, gradient_tensors)
    
    def get_requires_mini_batch(self):
        return True
    
# NOTE the reason why the grasp functions compute the scores and masks for a list of weights and grads and not a single one is because the results differ mathematically
# if we compute the scores for the layers individually vs together at once
# The grasp mask method could instead be made to individually create a mask per score tensor (only the score calculation is dependent between layers)

def tf_create_grasp_scores(weights, grads):
    grasp_scores = []

    # grad_dot = sum_i <g_i, g_i>
    grad_dot = tf.add_n([tf.reduce_sum(g * g) for g in grads])

    # Hessian-gradient product approximation
    # This causes the 2nd backward pass (gradient of gradient dot w.r.t. weights)
    Hg = tf.gradients(grad_dot, weights)

    # Grasp scores: -g * (Hg)
    for w, g, h in zip(weights, grads, Hg):
        grasp_scores.append(-g * h)

    return grasp_scores
    
def tf_create_grasp_masks(sparsity, weights, grads):
    
    grasp_scores = tf_create_grasp_scores(weights, grads)
    
    new_masks = []
    for i, w in enumerate(weights):
        score = grasp_scores[i]

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