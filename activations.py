import numpy as np

class Activation:
    def __init__(self):
        pass

    def forward(self, Z):
        pass

    def backward(self, dA, Z):
        pass

class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.maximum(0,Z)

    def backward(self, dA, Z):
        dZ = dA*(Z>0)
        return dZ
    
class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return np.maximum(Z*0.01, Z)
    
    def backward(self, dA, Z):
        dZ = dA*(Z > 0) + dA*(Z<=0)*0.01
        return dZ

class sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        return 1/(1+np.exp(-Z))

    def backward(self, dA, Z):
        s = sigmoid(Z)
        dZ = s*(1-s)
        return dA*dZ

class softmax(Activation):
    def __init__(self):
        super().__init__()

    def softmax(Z):
        stableZ = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(stableZ)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ/denominator

    def softmax_backward(dA, Z):
        pass