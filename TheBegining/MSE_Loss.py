import numpy as np 

def mse_loss(yTrue, yPred):
    return( (yTrue - yPred) ** 2).mean()

yTrue = np.array([1, 0, 0, 1])
yPred = np.array([0, 0, 0, 0])

print("MSE is: %f" %(mse_loss(yTrue, yPred)))