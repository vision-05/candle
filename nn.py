# A module to generalise neural network building
# With optimisations

import numpy as np
import activations
import transforms
import costs

# to add: batch gradient calculation

class nn:
    def __init__(self):
        self.fns = []

    def __repr__(self):
        """Debugging function, show the internal state"""
        prn = ""
        for fn in self.fns:
            prn += str(fn)
        return prn
    
    def model(self, *args):
        """Create the model for the neural network to train
           Adds activation functions and transforms to their respective lists IN ORDER"""
        for arg in args:
            if isinstance(arg, costs.cost):
                self.cost_fn = arg
            self.fns.append(arg)

        self.dJdA = lambda A, Y: A-Y #for 2 special cases not including linear regression

    def forward(self, X):
        res = X
        for fn in self.fns:
            res = fn.forward(res)

        return res

    def backward(self):
        pass

    def train(self):
        pass
