import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#add a ReLu Activation function exponentional
#add Normalization in it
#compute all of that on batch size
#minus the max values to prevent the overflow error

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

#print(np.sum(X, axis=1, keepdims=True))

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #axis and keepdims is to select the batch-size and keep the previous dims
        self.output = probabilities

#nnfs to call data
x, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_ReLU()

dense1.forward(x)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(x[:5])
print(activation2.output[:5])


