import numpy as np

class Softmax:
    # a fully-connected layer with softmax activation

    def __init__(self, inputSize, nodes):
        # devide by inputLength to reduce the variance of initial values
        self.weights = np.random(inputSize, nodes) / inputSize
        self.biases = np.zeros(nodes)  # set all biases to zero

    def forward(self, input):
        # Returns a 1d array containing the respective probability values.

        """
            flatten() the input to make it easier to work with, since 
            we no longer need its shape.
        """
        input = input.flatten()

        total = np.dot(input, self.weights) + self.biases
        exp = np.exp(total)

        return exp/np.sum(exp, axis = 0)