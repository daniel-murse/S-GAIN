"""
A DST strategy that uses independent generator and discriminator strategies, splitting strategies into 3 components:
 - initialisation of the mask
 - pruning of the mask
 - regrowing of the mask
"""

from strategies.strategy import Strategy


class SimpleStrategy(Strategy):

    def start_train(self, *args, **kwargs):
        return super().start_train(*args, **kwargs)
    
    def iteration(self, *args, **kwargs):
        return super().iteration(*args, **kwargs)
    
    def end_train(self, *args, **kwargs):
        return super().end_train(*args, **kwargs)

def create_simple_strategy():
    pass