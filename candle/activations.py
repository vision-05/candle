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

"""Dense (Linear) activation (affine) layer with forward and backward propogation"""
class Dense(Activation):
    def __init__(self, in_features, out_features, bias=True):
        """Initialize weights and biases with He initialization, taking *in_features* and *out_features* as input and output dimensions respectively."""
        super().__init__()
        self.bias_pred = True
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.biases = np.ones((1,out_features)) * 0.01

        self.A_prev = None
        self.dW = None
        self.dB = None

    def forward(self, inputs):
        """Perform forward propogation as Wx + b."""
        self.A_prev = inputs
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(inputs, self.weights) + self.biases
    
    def predict(self, inputs):
        """Stateless forward propogation for inference."""
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dZ, **kwargs):
        """Stateful backward propogation returning gradient of this layer's inputs. Gradients of weights and biases are stored in self.dW and self.dB respectively."""
        m = self.A_prev.shape[0]

        lambd = kwargs.get("lambd", 0.0)

        L2_grad = lambd*self.weights/m

        self.dW = np.dot(self.A_prev.T, dZ)/m + L2_grad
        self.dB = np.sum(dZ, axis=0, keepdims=True)/m

        new_grad = np.dot(dZ, self.weights.T)
        return new_grad
    
class Conv1D(Activation):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        """1 Dimenional Convolutional layer"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel_size) * np.sqrt(2 / self.in_channels+self.kernel_size)
        
        self.biases = np.zeros(out_channels)

    def im2col(self, X):
        pass

    def forward(self, inputs):
        weights_flattened = self.weights.reshape(self.out_channels, -1) #combine channels with kernels
        inputs_padded = np.pad(inputs, self.padding, 'constant', constant_values=0) #pad inputs
        return np.dot(weights_flattened, inputs_padded) + self.biases[:,np.newaxis]

    def backward(self, dZ, **kwargs):
        """Stateful backpropogation"""
        m = self.A_prev.shape[1]

        lambd = kwargs.get("lambd", 0.0)

        L2_grad = lambd*self.weights/m

        self.dW = np.zeros_like(self.weights)
        self.dB = np.sum(dZ, axis=1, keepdims=True)

        new_grad = np.dot(dZ, self.weights.T)
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