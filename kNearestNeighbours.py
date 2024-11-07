from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import dataHandling as dh

# K Nearest Neighbours
def KNN(database, num):
    # Loading all data from database using load_data which takes 2 arguments: the database and a limitation of the dataset
    # should we want one for execution speed. If we want the whole dataset, we leave the limit as 0.
    input_data, input_labels, test_data, test_labels = dh.load_data(database, 0)

    # K nearest neighbours class that takes the desired number of neighbours as an argument.
    knn = KNeighborsClassifier(num)
    knn.fit(input_data, input_labels)

    prediction = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, prediction)
    print(f"Accuracy with K-Nearest Neighbours with k={num}: {accuracy * 100:.2f}%")
    return accuracy
