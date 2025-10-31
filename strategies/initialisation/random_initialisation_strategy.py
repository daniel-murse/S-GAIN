"""
Mask initialisation strategy that randomly masks weights.
"""

import numpy as np
from strategies.initialisation.initialisation_strategy import InitialisationStrategy

class RandomInitialisationStrategy(InitialisationStrategy):
    def __init__(self, sparsity):
        super().__init__()
        self.sparsity = sparsity

    def get_tf_mask_initialisation_tensors(self, weight_tensors):
        mask_tensors = [tf_mask_init_random(self.sparsity, w) for w in weight_tensors]
        return mask_tensors
    
def tf_mask_init_random(sparsity, weight_tensor):
    # NOTE HACK tf_random_mask_init its ok to return a numpy object because tf will cast numpy to a tensor when used in operations
    # NOTE is random, so it may not keep the exact sparsity (+- a few percent)
    mask_tensor = np.random.choice([0, 1], size=weight_tensor.shape, p=[sparsity, 1 - sparsity])
    return mask_tensor