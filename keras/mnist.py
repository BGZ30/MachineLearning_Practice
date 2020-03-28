import numpy as np
import mnist_loader

def data():
    return mnist_loader.load_data()

if __name__ == "__main__":
    training_set, validation_set, test_set = mnist_loader.load_data()
   
    print(training_set[0].shape) 
    print(training_set[1].shape)

    print(test_set[0].shape) 
    print(test_set[1].shape) 