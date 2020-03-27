"""
    This is a simple neural network
    Using Stocastic Gradient Descent SGD

    The training process will look like this:
        1- Choose one sample from our dataset. This is what makes it stochastic gradient descent - we only operate on one sample at a time.
        2- Calculate all the partial derivatives of loss with respect to weights or biases
        3- Use the update equation to update each weight and bias.
        4- Go back to step 1.
"""

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derevative of the segmoid function
def derivSigmoid(x):
    # Derevative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx*(1-fx)

# The Loss function: MSE
def mse_loss(yTrue, yPred):
    return( (yTrue - yPred) ** 2).mean()

class NeuralNetwork:
    """
    A neural network with:
        -  2 inputs.
        -  a hidden layer with 2 neurons.
        -  an output layer with single neuron.
    Weights and biases are random
    """
    def __init__(self):
        # Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        
        # Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    
    def feedForward(self, x):
        n1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        n2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o = sigmoid(self.w5 * n1 + self.w6 * n2 + self.b3)

        return o

    def train(self, data, trueYs):
        """
            data is (n x 2) array, n is the number of samples
        """
        learning_rate = 0.1
        epochs = 1000 # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, yTrue in zip(data, trueYs):
                sum_n1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                n1 = sigmoid(sum_n1)

                sum_n2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                n2 = sigmoid(sum_n2)

                sum_o = self.w5 * n1 + self.w6 * n2 + self.b3
                o = sigmoid(sum_o)

                yPred = o

                # Calculating partial derevatives
                dL_dYPred = -2 * (yTrue - yPred)

                # Neuron o
                dYPred_dW5 = n1 * derivSigmoid(sum_n1)
                dYPred_dW6 = n2 * derivSigmoid(sum_n2)
                dYPred_dB3 =  derivSigmoid(sum_o)

                dYPred_dN1 = self.w5 * derivSigmoid(sum_o)
                dYPred_dN2 = self.w6 * derivSigmoid(sum_o)

                # Neuron n1
                dN1_dW1 = x[0] * derivSigmoid(sum_n1)
                dN1_dW2 = x[1] * derivSigmoid(sum_n1)
                dN1_dB1 =  derivSigmoid(sum_n1)

                # Neuron n2
                dN2_dW3 = x[0] * derivSigmoid(sum_n2)
                dN2_dW4 = x[1] * derivSigmoid(sum_n2)
                dN2_dB2 =  derivSigmoid(sum_n2)

                # Update the weights and biases
                # Neuron n1
                self.w1 -= learning_rate * dL_dYPred * dYPred_dN1 * dN1_dW1
                self.w2 -= learning_rate * dL_dYPred * dYPred_dN1 * dN1_dW2
                self.b1 -= learning_rate * dL_dYPred * dYPred_dN1 * dN1_dB1

                # Neuron n2
                self.w3 -= learning_rate * dL_dYPred * dYPred_dN2 * dN2_dW3
                self.w4 -= learning_rate * dL_dYPred * dYPred_dN2 * dN2_dW4
                self.b2 -= learning_rate * dL_dYPred * dYPred_dN2 * dN2_dB2 

                # Neuron o
                self.w5 -= learning_rate * dL_dYPred * dYPred_dW5
                self.w6 -= learning_rate * dL_dYPred * dYPred_dW6
                self.b3 -= learning_rate * dL_dYPred * dYPred_dB3

                # total loss at the end of each epoch
                if epoch % 10 == 0:
                    predYs = np.apply_along_axis(self.feedForward, 1, data)
                    loss = mse_loss(trueYs, predYs)

                    print("Epoch %d loss: %.3f" %(epoch, loss))

# Define dataset
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-16, -6],
])

trueYs = np.array([
    1,
    0,
    0,
    1,
])

# Start Training
network = NeuralNetwork()
network.train(data, trueYs)

# Start Testing
mathew = np.array([50, 6])
jessey = np.array([-8, -4])

print("mathew is: %.3f" %(network.feedForward(mathew)))
print("jessey is: %.3f" %(network.feedForward(jessey)))                 