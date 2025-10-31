import tensorflow as tf
from strategies.pruning.prune_strategy import PruneStrategy

# The tf_mask_prune_magnitude_[count, fraction] are 2 different functions because although they are similar, reusing the count one for the fraction can hurt performance (even though tf may optimise identical trees in nodes)

def tf_mask_prune_magnitude_count(mask_tensor, weight_tensor, prune_count, seed=None):
    # To approach this in tf, we need to flatten the mask, update it with scattered updates (update a flat list in multiple indices)
    # Then we will reshape the flat list to the original mask shape and we have the updated mask
    # We will sort the flat weights and find the corresponding smallest value at prune_count and then mask out weights less than or equal to it
    # NOTE HACK this may prune more elements than prune_count if there are many elements equal to the threshold (extremely unlikely). Change to tf.equal if desired, but this can have other sideffects
    # In practice it doesnt matter as the chance any 2 weights are equal is very small, and the chance there are many weights at exactly threshold to significantly disrupt the sparsity is even smaller (astronomically small)

    weight_tensor_flat = tf.reshape(weight_tensor, [-1])
    mask_tensor_flat = tf.reshape(mask_tensor, [-1])

    # Masked weights (only consider active ones)
    # This gets us the values, not the indices (tf.boolean_mask instead of tf.where)
    active_weights = tf.boolean_mask(weight_tensor_flat, tf.equal(mask_tensor_flat, 1))
    active_count = tf.shape(active_weights)[0]
    prune_count = tf.minimum(prune_count, active_count)

    # Get the magnitudes (absolute values) and sort ascending
    abs_active_weights = tf.abs(active_weights)
    # tf sort: the sort is ascending so we negative the weights to find the smallest in magnitude (as its closer to 0 so its bigger in the sort)
    threshold = tf.nn.top_k(-abs_active_weights, k=prune_count).values[-1] * -1

    # Node for pruning: the entry in the mask was 1 originally in the mask or is bigger than the threshold of the prune_count(th) smallest element
    # This is corresponding element wise in the flat lists
    should_prune = tf.logical_and(tf.equal(mask_tensor_flat, 1),
                                  tf.less_equal(tf.abs(weight_tensor_flat), threshold))

    # Use the other overload of tf.where; we wont get indices, we will combine the updated mask and the old mask (read the docs of tf.where if you dont know)
    new_mask_flat = tf.where(should_prune, tf.zeros_like(mask_tensor_flat), mask_tensor_flat)

    # Return the reshaped flat new mask
    return tf.reshape(new_mask_flat, tf.shape(mask_tensor))

def tf_mask_prune_magnitude_fraction(mask_tensor, weight_tensor, fraction, seed=None):
    # To approach this in tf, we need to flatten the mask, update it with scattered updates (update a flat list in multiple indices)
    # Then we will reshape the flat list to the original mask shape and we have the updated mask
    # We will sort the flat weights and find the corresponding smallest value at prune_count and then mask out weights less than or equal to it
    # NOTE HACK this may prune more elements than prune_count if there are many elements equal to the threshold (extremely unlikely). Change to tf.equal if desired, but this can have other sideffects
    # In practice it doesnt matter as the chance any 2 weights are equal is very small, and the chance there are many weights at exactly threshold to significantly disrupt the sparsity is even smaller (astronomically small)

    weight_tensor_flat = tf.reshape(weight_tensor, [-1])
    mask_tensor_flat = tf.reshape(mask_tensor, [-1])

    # Masked weights (only consider active ones)
    # This gets us the values, not the indices (tf.boolean_mask instead of tf.where)
    active_weights = tf.boolean_mask(weight_tensor_flat, tf.equal(mask_tensor_flat, 1))
    active_count = tf.shape(active_weights)[0]
    # Get the prune count by the fraction of active weights
    prune_count = tf.cast(tf.round(fraction * tf.cast(active_count, tf.float32)), tf.int32)
    prune_count = tf.minimum(prune_count, active_count)

    # Get the magnitudes (absolute values) and sort ascending
    abs_active_weights = tf.abs(active_weights)
    # tf sort: the sort is ascending so we negative the weights to find the smallest in magnitude (as its closer to 0 so its bigger in the sort)
    threshold = tf.nn.top_k(-abs_active_weights, k=prune_count).values[-1] * -1

    # Node for pruning: the entry in the mask was 1 originally in the mask or is bigger than the threshold of the prune_count(th) smallest element
    # This is corresponding element wise in the flat lists
    should_prune = tf.logical_and(tf.equal(mask_tensor_flat, 1),
                                  tf.less_equal(tf.abs(weight_tensor_flat), threshold))

    # Use the other overload of tf.where; we wont get indices, we will combine the updated mask and the old mask (read the docs of tf.where if you dont know)
    new_mask_flat = tf.where(should_prune, tf.zeros_like(mask_tensor_flat), mask_tensor_flat)

    # Return the reshaped flat new mask
    return tf.reshape(new_mask_flat, tf.shape(mask_tensor))


class MagnitudePruneStrategy(PruneStrategy):
    def __init__(self, prune_count_func):
        super().__init__()
        self.prune_count_func = prune_count_func

    def get_tf_pruned_mask_tensors(self, training_loop_iteration, weight_tensors, mask_tensors):
        prune_counts = self.prune_count_func(training_loop_iteration)
        if prune_counts is not None:
            return [tf_mask_prune_magnitude_count(m, w, pc) for (m, w), pc in zip(zip(mask_tensors, weight_tensors), prune_counts) if pc is not None]
        return None