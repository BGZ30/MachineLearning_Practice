import pickle
import gzip

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    images = np.concatenate((training_set[0], validation_set[0]))
    lables = np.concatenate((training_set[1], validation_set[1]))
    training_set = (images, lables)

    f.close()
    return (training_data, validation_data, test_data)