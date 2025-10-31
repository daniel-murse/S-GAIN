"""
Mask initialisation strategy that masks the smallest weights (prune by magnitude).
"""

import tensorflow as tf
from strategies.initialisation.initialisation_strategy import InitialisationStrategy

class MagnitudeInitialisationStrategy(InitialisationStrategy):
    def __init__(self, sparsity):
           super().__init__()
           self.sparsity = sparsity

    def get_tf_mask_initialisation_tensors(self, weight_tensors):
        # We will return a parallel list to the tf weight tensors representing corresponding tf mask tensors
        mask_tensors = [tf_mask_init_magnitude(self.sparsity, weight_tensor) for weight_tensor in weight_tensors]
        return mask_tensors

def tf_mask_init_magnitude(sparsity, weight_tensor):
    # The magnitudes in the weight tensor, abs them to sort by magnitude
    magnitudes = tf.abs(weight_tensor)
    # Get the values flat for sorting
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
    mask_tensor = tf.cast(tf.greater(magnitudes, threshold), weight_tensor.dtype)
    
    return mask_tensor