import tensorflow as tf
from strategies.pruning.prune_strategy import PruneStrategy

# Tthe tf_mask_prune_random_[count, fraction] are 2 different functions because although they are similar, reusing the count one for the fraction can hurt performance (even though tf may optimise identical trees in nodes)

def tf_mask_prune_random_count(mask_tensor, prune_count, seed=None):
    # List of the indices of non zero mask entries (active weights)
    active_indices = tf.where(tf.equal(mask_tensor, 1))
    # Count the number of active weights to clip the prune count safely (this is the dimension of the list of ones indices)
    # We need it as a tf node
    active_count = tf.shape(active_indices)[0]
    # Clip/bound the prune count
    prune_count = tf.minimum(prune_count, active_count)

    # Randomly select indices of active weights to prune

    # We do this by shuffling the active indices, then selecting from shuffled list in one block
    shuffled_ones_indices = tf.random.shuffle(active_indices, seed=seed)

    # This is the indices to prune
    prune_indices = shuffled_ones_indices[:prune_count]

    # This is the updated flat list with zeros in the prune indices
    mask_updates = tf.tensor_scatter_nd_update(mask_tensor,
                                               prune_indices,
                                               tf.zeros([prune_count], dtype=mask_tensor.dtype))
    
    # Reshape the flat list to the original mask shap
    return mask_updates

def tf_mask_prune_random_fraction(mask_tensor, fraction, seed=None):
    # Indices of non zero mask entries (active weights)
    active_indices = tf.where(tf.equal(mask_tensor, 1))
    # Count the number of active weights to clip the prune count safely (this is the dimension of the list of ones indices)
    # We need it as a tf node
    active_count = tf.shape(active_indices)[0]
    # Clip/bound the prune count
    prune_count = tf.cast(tf.round(fraction * tf.cast(active_count, tf.float32)), tf.int32)
    prune_count = tf.minimum(prune_count, active_count)

    # Randomly select indices of active weights to prune

    # We do this by shuffling the active indices, then selecting from shuffled list in one block
    shuffled_ones_indices = tf.random.shuffle(active_indices, seed=seed)

    # This is the indices to prune
    prune_indices = shuffled_ones_indices[:prune_count]

    # This is the updated flat list with zeros in the prune indices
    mask_updates = tf.tensor_scatter_nd_update(mask_tensor,
                                               prune_indices,
                                               tf.zeros([prune_count], dtype=mask_tensor.dtype))
    
    # Reshape the flat list to the original mask shap
    return mask_updates


class RandomPruneStrategy(PruneStrategy):
    def __init__(self, prune_count_func):
        super().__init__()
        self.prune_count_func = prune_count_func

    def get_tf_pruned_mask_tensors(self, training_loop_iteration, weight_tensors, mask_tensors):
        prune_counts = self.prune_count_func(training_loop_iteration)
        if prune_counts is not None:
            print("rpune", prune_counts)
            return [tf_mask_prune_random_count(m, pc) for m, pc in zip(mask_tensors, prune_counts) if pc is not None]
        return None