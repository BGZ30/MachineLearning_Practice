import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedForward(self, inpus):
        # w.x + b
        total = np.dot(self.weights, inpus) + self.bias

        return sigmoid(total)


class NeuralNetwork:
    """
    A neural network with:
        -  2 inputs.
        -  a hidden layer with 2 neurons.
        -  an output layer with single neuron.
    All neurons have the same weights and bias
    """
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # define the neurons of the hidden layer
        self.n1 = Neuron(weights, bias)
        self.n2 = Neuron(weights, bias)

        # define the output layer
        self.out = Neuron(weights, bias)

    def feedForward(self, x):
        out_n1 = self.n1.feedForward(x)
        out_n2 = self.n2.feedForward(x)

        # pass those as inputs for the out neuron
        input4out = np.array([out_n1, out_n2]) 
        finalOut = self.out.feedForward(input4out)

        return finalOut

network = NeuralNetwork()
x = np.array([2, 3])

print("The output of this network is: %f" %(network.feedForward(x)))