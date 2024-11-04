import pickle
import os
import numpy as np

# As mentioned in readme.html, each of these files is a Python "pickled" object produced with cPickle,
# so, we "unpickle" them accodringly.
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')  # Specify 'latin1' to avoid decoding issues : Recommended by ChatGPT.
    return dict

# Load all data batches and limits the dataset for execution speed if the limit value is set to one higher than 0.
def load_data(data_dir, limit):
    # Axis array initialisation.
    x_train = []
    y_train = []
    
    # If limit is 0, it will be interpreted as false and not limit the dataset,
    # if limit id higher than 0, it will limit the dataset acccordingly.
    

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

    if limit>0:
        olen=len(x_train)
        x_train, y_train, x_test, y_test = limit_dataset(limit, x_train, y_train, x_test, y_test)
        print("Limiting dataset from ",olen," to ", len(x_train))
    
    return x_train, y_train, x_test, y_test

# This function is mainly for test purposes; It limits the training datasets to a set number (num) and the testing
# datasets accordingly as in the database for every 5 training images there is 1 for testing.
def limit_dataset(num, x_train, y_train, x_test, y_test):
    x_train, y_train = x_train[:num], y_train[:num]  
    x_test, y_test = x_test[:num//5], y_test[:num//5]
    return x_train, y_train, x_test, y_test