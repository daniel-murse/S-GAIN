"""
Base class for a sparse training mask initialisation strategy for the S-GAIN training loop.

The idea behind the design of this API is a functional programming paradigm.
The base class acts as a functional interface, where instances of derived classes take parameters at construction
to produce a pure function. Derived implementations could/should expose helper functions they use for the strategy,
so that the implementation of the strategy is cleanly written and remains modular.
"""

class InitialisationStrategy:
    def get_tf_mask_initialisation_tensors(self, weight_tensors, gradient_tensors = None):
        """
        Returns a parallel list to the tf weight tensors, which contains tf nodes that represent initialisation for masks for the weight tensors.
        """
        raise NotImplementedError
    
    def get_requires_mini_batch(self):
        """
        Boolean whether the initialisation strategy requires a minibatch for computation.
        """
        raise NotImplementedError