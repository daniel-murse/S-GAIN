
import tensorflow as tf
from strategies.regrowing.regrow_strategy import RegrowStrategy

def tf_mask_and_weight_regrow_randomly_normal_xavier_init_count(weights, mask, regrow_count, seed=None):
    inactive_indices = tf.where(tf.equal(mask, 0))
    inactive_count = tf.shape(inactive_indices)[0]

    regrow_count = tf.minimum(regrow_count, inactive_count)
    shuffled_indices = tf.random.shuffle(inactive_indices, seed=seed)
    regrow_indices = shuffled_indices[:regrow_count]

    # normal xavier computation parameters in tf 
    weight_shape = tf.shape(weights)
    weight_rank = tf.rank(weights)
    fan_in = tf.cond(tf.equal(weight_rank, 2),
                     lambda: tf.cast(weight_shape[0], tf.float32),
                     lambda: tf.cast(tf.reduce_prod(weight_shape[:-1]), tf.float32))
    fan_out = tf.cast(weight_shape[-1], tf.float32)
    limit = tf.sqrt(6.0 / (fan_in + fan_out))

    # Random new weights for regrown connections. One for each regrow index
    regrow_weights = tf.random.uniform([regrow_count], -limit, limit, dtype=weights.dtype.base_dtype, seed=seed)

    # Update weights and mask by scattering updates to the old weights and mask with the regrown indices and values
    new_weights = tf.tensor_scatter_nd_update(weights, regrow_indices, regrow_weights)
    new_mask = tf.tensor_scatter_nd_update(mask, regrow_indices, tf.ones([regrow_count], dtype=mask.dtype))

    return new_weights, new_mask

class RandomNormalXavierRegrowStrategy(RegrowStrategy):
    def __init__(self, regrow_count_func):
        super().__init__()
        self.regrow_count_func = regrow_count_func

    def get_tf_regrowed_mask_and_weight_tensors(self, training_loop_iteration, weight_tensors, mask_tensors):
        regrow_counts = self.regrow_count_func(training_loop_iteration)
        if(regrow_counts is not None):

            return [tf_mask_and_weight_regrow_randomly_normal_xavier_init_count(w, m, rc) for (w, m), rc in zip(zip(weight_tensors, mask_tensors), regrow_counts) if rc is not None]
        return None