# train MNIST dataset using CNN from scratch

from cnnScratch import Layer
from cnnScratch import Dense , Convolutional
from cnnScratch import Activation , Reshape
from cnnScratch import mse, mse_prime , binary_cross_entropy, binary_cross_entropy_prime
from cnnScratch import Sigmoid , Tanh

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

#limit data on binary 1,0 digits and limit our database 
def process_data(x, y, limit):
    zero_index = np.where(y==0)[0][:limit]
    one_index = np.where(y==0)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices,], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32")/255
    y = np_utils.to_categorical(y)
    print(y.shape)
    y = y.reshape(len(y)//2, 2, 1)

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = process_data(x_train, y_train, 100)
x_test, y_test = process_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += binary_cross_entropy(y, output)

        # backward
        grad = binary_cross_entropy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error={error}")

# test
for x, y in zip(x_test, y_test):
    output = x
    for layer in network:
        output = layer.forward(output)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")