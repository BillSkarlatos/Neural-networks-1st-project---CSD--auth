import pickle
import os
import numpy as np
# As mentioned in readme.html, each of these files is a Python "pickled" object produced with cPickle.
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')  # Specify 'latin1' to avoid decoding issues : Recommended by ChatGPT.
    return dict

# Load all data batches
def load_data(data_dir):
    # Axis array initialisation.
    x_train = []
    y_train = []
    
    # CIFAR-10 has 5 training batches and 1 test batch
    for i in range(1, 6):
        data_dict = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(data_dict['data'])
        y_train += data_dict['labels']
    
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    
    # Load test batch
    test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    x_test = test_dict['data']
    y_test = np.array(test_dict['labels'])
    
    return x_train, y_train, x_test, y_test