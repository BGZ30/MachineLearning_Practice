import numpy as np
import pickle
import gzip

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    images = np.concatenate((training_data[0], validation_data[0]))
    lables = np.concatenate((training_data[1], validation_data[1]))
    training_data = (images, lables)

    f.close()
    # return (training_data, validation_data, test_data)
    return (training_data, test_data)