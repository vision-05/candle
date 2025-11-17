"""Activation functions for neural networks."""

import numpy as np

"""Base Activation class and abstract forward and backward methods."""
class Activation:
    def __init__(self):
        pass

    def forward(self, Z):
        pass

    def backward(self, dA, Z, **kwargs):
        pass

"""Linear activation (affine) layer with forward and backward propogation"""
class Linear(Activation):
    def __init__(self, in_features, out_features, bias=True):
        """Initialize weights and biases with He initialization, taking *in_features* and *out_features* as input and output dimensions respectively."""
        super().__init__()
        self.bias_pred = True
        self.weights = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        self.biases = np.ones((out_features, 1)) * 0.01

        self.A_prev = None
        self.dW = None
        self.dB = None

    def forward(self, inputs):
        """Perform forward propogation as Wx + b."""
        self.A_prev = inputs
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(self.weights, inputs) + self.biases
    
    def predict(self, inputs):
        """Stateless forward propogation for inference."""
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(self.weights, inputs) + self.biases
    
    def backward(self, dZ, **kwargs):
        """Stateful backward propogation returning gradient of this layer's inputs. Gradients of weights and biases are stored in self.dW and self.dB respectively."""
        m = self.A_prev.shape[1]

        lambd = kwargs.get("lambd", 0.0)

        L2_grad = lambd*self.weights/m

        self.dW = np.dot(dZ, self.A_prev.T)/m + L2_grad
        self.dB = np.sum(dZ, axis=1, keepdims=True)/m

        new_grad = np.dot(self.weights.T, dZ)
        return new_grad

"""ReLU activation function with forward and backward propogation."""
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        """Apply ReLU activation function as max(0, Z)."""
        self.Z = Z
        return np.maximum(0,Z)
    
    def predict(self, Z):
        """Stateless ReLU activation for inference."""
        return np.maximum(0,Z)

    def backward(self, dZ, **kwargs):
        """Compute gradient of ReLU activation during backpropagation."""
        new_grad = dZ*(self.Z>0)
        return new_grad
    
"""Leaky ReLU activation function with forward and backward propogation."""
class LeakyReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        """Apply Leaky ReLU activation function as max(0.01*Z, Z)."""
        self.Z = Z
        return np.maximum(Z*0.01, Z)
    
    def predict(self, Z):
        """Stateless Leaky ReLU activation for inference."""
        return np.maximum(Z*0.01, Z)
    
    def backward(self, dZ, **kwargs):
        """Compute gradient of Leaky ReLU activation during backpropagation."""
        new_grad = dZ*(self.Z > 0) + dZ*(self.Z<=0)*0.01
        return new_grad

"""Sigmoid activation function with forward and backward propogation."""
class sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        """Apply Sigmoid activation function as 1/(1+exp(-Z))."""
        self.Z = Z
        return 1/(1+np.exp(-Z))
    
    def predict(self, Z):
        """Stateless Sigmoid activation for inference."""
        return 1/(1+np.exp(-Z))

    def backward(self, dZ, **kwargs):
        """Compute gradient of Sigmoid activation during backpropagation."""
        s = sigmoid(self.Z)
        return dZ*s*(1-s)

"""Softmax activation function with forward and backward propogation."""
class softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        """Apply Softmax activation function as exp(Z)/sum(exp(Z)), using numerical stability tricks to avoid overflow and explosion."""
        self.Z = Z
        stableZ = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(stableZ)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ/denominator
    
    def predict(self, Z):
        """Stateless Softmax activation for inference."""
        stableZ = Z - np.max(Z, axis=0, keepdims=True)
        expZ = np.exp(stableZ)
        denominator = np.sum(expZ, axis=0, keepdims=True)
        return expZ/denominator

    def backward(self, dZ, **kwargs):
        """Compute gradient of Softmax activation during backpropagation."""
        pass