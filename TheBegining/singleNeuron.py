import numpy as np

# define the activation function
def sigmoid(x):
    # f(x) =  / ( 1+ e^(-x) )
    return 1/(1 + np.exp(-x))

# build the neuron
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # w.x + b
        total = np.dot(self.weights, inputs) + self.bias

        # pass that to the activation function
        return sigmoid(total)

weights = np.array([0, 1])
bias = 4

# neuron
n = Neuron(weights, bias) 

# features
x = np.array([2, 3])

print("The output of this neuron is: %f" %(n.feedforward(x)))