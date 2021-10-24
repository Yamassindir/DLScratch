inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
biais = 3

output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+biais


#
inputs = [1.2, 5.1, 2.1, 1]

weights1 = [3,0.2, 0.1, 8.7]
weights2 =  [3,0.2, 0.1, 8.7]
weights3 =  [3,0.2, 0.1, 8.7]

biais1 = 3
biais2 = 1
biais3 = 0.2

output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+biais1,
         inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+biais2,
        inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+biais3]


#writter better
inputs = [1.2, 5.1, 2.1, 1]

weights = [[3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7]]

biais = [3, 0.2, 1]

layer_outputs = [] #output of the current layer
for neuron_weights, neuron_bais in zip(weights,biais):
    neuron_output = 0 #output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bais
    layer_outputs.append(neuron_output)


#more suffisticated way using numpy matrix operations
import numpy as np

outputs = np.dot(weights, inputs) + biais

#when inputs and weights have the same dimension
inputs = [[1.2, 5.1, 2.1, 1],
        [1.2, 5.1, 2.1, 1],
        [1.2, 5.1, 2.1, 1]]

weights = [[3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7]]

biais = [3, 0.2, 1]

outputs = np.dot(inputs, np.array(weights).T) + biais


#generalize for 2 layers
inputs = [[1.2, 5.1, 2.1, 1],
        [1.2, 5.1, 2.1, 1],
        [1.2, 5.1, 2.1, 1]]

weights1 = [[3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7],
           [3,0.2, 0.1, 8.7]]

biais1 = [3, 0.2, 1]


weights2 = [[3,0.2, 0.1],
           [3,0.2, 0.1],
           [3,0.2, 0.1]]

biais2 = [3, 0.2, 1]

layer1_outputs = np.dot(inputs, np.array(weights1).T) + biais1


#the inputs for layer2 are the outputs of the 1st one
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biais2



#in a class now
import numpy as np 

np.random.seed(0)
X = [[2, 5, 1, 1],
    [1, 1, 2, 1],
    [3.2, 0.1, 1.1, 4]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
#print(layer1.outputs)
layer2.forward(layer1.outputs)
print(layer2.outputs)

