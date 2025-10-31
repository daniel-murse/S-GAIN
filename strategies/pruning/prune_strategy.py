"""
Base class for a mask/weight pruning strategy for the S-GAIN training loop.
"""

class PruneStrategy:
    def get_tf_pruned_mask_tensors(self, training_loop_iteration, weight_tensors, mask_tensors):
        """
        Returns a parallel list to the tf weight tensors, which contains tf nodes that represent initialisation for masks for the weight tensors.
        If a mask should be re-applied, it should be not None, else None.
        """
        raise NotImplementedError