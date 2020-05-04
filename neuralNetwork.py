import numpy as np

# FIXME: Set seed for shared experience, this is study only!
np.random.seed(0)

# Input data, 3 samples
# Should be scaled into numbers between -1 and 1
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Hidden layers (not handled by programmer)
# Weights should be initialized between -1 and 1
# Biases should start as zero, except in the case of 'dead network' 0s
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights already transposed
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # np.zeros needs a tuple of the shape
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

'''
# Inputs * weights + bias
Basic:
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

As Loop:
layer_outputs = [] # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 # Output given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

With Numpy and dot Product:
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

'''

'''
###################NOTES##################
Shape: The shape of the array, must be homologous
    ie: lolol = [[[4,2,5,6],
                  [4,2,6,1]],
                 [[6,9,1,5],
                  [6,4,8,4]],
                 [[2,8,5,3],
                  [1,1,9,4]]]
        the shape of lolol is (3,2,4)
                {3 arrays of 2 vectors of 4 elements} (3D array)
            A shape with two elements is a matrix, shape with 1 element is a vector
Tensor: an object that can be represented as an array
Dot_Product: a=[1,2,3] b=[2,3,4] dot_product = a[0]*b[0] + a[1] * b[1]...
Batches: Helps with parallelization and generalization of fitment. (32 - 64)
**All numbers should tend to numbers between -1 and 1
'''
