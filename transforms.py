import numpy as np

class Transform:
    def __init__(self):
        pass

    def forward(self):
        pass

class Linear(Transform):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.bias_pred = True
        self.weights = np.zeros((out_features, in_features))
        self.biases = np.zeros((out_features, 1))

    def forward(self, inputs):
        self.biases = 10
        self.weights = 20
        if self.bias_pred == False:
            self.bias = 0

        return np.dot(self.weights, inputs) + self.bias