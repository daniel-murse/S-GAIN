"""Interface for a DST strategy, for static typing."""

class Strategy:
    def start_train(self, *args, **kwargs):
        """
        Called by the S-GAIN training loop before the training loop. Static sparsity calculations can be done here.
        """
        raise NotImplementedError

    def iteration(self, *args, **kwargs):
        """
        Called by the S-GAIN training loop during training, before sparsity metric calculations and the forward pass.
        The mask should be enforced here. Strategies should keep track of the iteration count themselves.
        """
        raise NotImplementedError
    
    def end_train(self, *args, **kwargs):
        """
        Called by the S-GAIN training loop after training. Weights will have recieved one last update from backpropagation,
        so enforcing the masks once more here is heavily recommended to keep the sparsity in the final model.
        """
        raise NotImplementedError