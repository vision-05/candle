# A module to generalise neural network building
# With optimisations

import numpy as np
import activations
import costs

# to add: batch gradient calculation
# OOM from keeping too many parameters

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
            if isinstance(arg, costs.Cost):
                self.cost_fn = arg
            else:
                self.fns.append(arg)

        self.dJdZ = lambda A, Y: A-Y #for 2 special cases not including linear regression

    def forward(self, X):
        res = X
        for fn in self.fns:
            res = fn.forward(res)

        return res
    
    def predict(self, X):
        res = X
        for fn in self.fns:
            res = fn.predict(res)
        return res

    def backward(self, Y_pred, Y):
        dZ = self.dJdZ(Y_pred, Y)
        for fn in reversed(self.fns[:-1]): #first grad already calculated
            dZ = fn.backward(dZ)

    def accuracy(self, Y_pred, Y):
        predicted_labels = np.argmax(Y_pred, axis=0)
        true_labels = np.argmax(Y, axis=0)
        correct = (predicted_labels == true_labels)
        accuracy_val = np.mean(correct)
        return accuracy_val

    def train(self, X, Y, X_val, Y_val, no_epochs, learning_rate):
        print("Training")
        batch_size = 500

        m = X.shape[1]
        for i in range(no_epochs):
            epoch_cost = 0
            permutation = np.random.permutation(m)

            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            no_batches = m//batch_size

            for j in range(no_batches):
                batch_X = X[:,batch_size*j:batch_size*(j+1)]
                batch_Y = Y[:,batch_size*j:batch_size*(j+1)]
                batch_Y_pred = self.forward(batch_X)

                cost = self.cost_fn.compute_cost(batch_Y_pred, batch_Y)
                epoch_cost += cost

                self.backward(batch_Y_pred, batch_Y)

                for fn in self.fns:
                    if isinstance(fn, activations.Linear):
                        fn.weights -= learning_rate * fn.dW
                        fn.biases -= learning_rate * fn.dB

            if i % 10 == 0 or i == no_epochs - 1:
                Y_val_pred = self.predict(X_val)
                Y_train_pred = self.predict(X)
                Y_train_acc = self.accuracy(Y_train_pred, Y)
                Y_val_cost = self.cost_fn.compute_cost(Y_val_pred, Y_val)
                Y_val_acc = self.accuracy(Y_val_pred, Y_val)
                print(f"Epoch {i}: training cost {epoch_cost/no_batches}, validation cost {Y_val_cost}")
                print(f"Validation accuracy {Y_val_acc*100:.2f}%, training Accuracy {Y_train_acc*100:.2f}%")

