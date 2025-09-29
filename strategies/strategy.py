"""SAM interface/functional interface for a DST strategy, for static typing."""

class Strategy:
    def iteration(self, *args, **kwargs):
        raise NotImplementedError
    
    def end_train(self, *args, **kwargs):
        raise NotImplementedError