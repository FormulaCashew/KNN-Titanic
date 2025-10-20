from collections import Counter

import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances


class KNN:
    """
    Class KNN_Model with methods to implement KNN_Model

    """

    def __init__(self, k: int = 3):
        """
        Constructor of KNN_Model
        Args:
            k (int): number of neighbors to use, defaults to 3, it is recommended to use an odd number
        """
        self._outputs_train = None
        self._inputs_train = None
        self.k = k

    def store(self, inputs_train, outputs_train):
        """
        Stores training data
        Args:
            inputs_train (pandas.DataFrame): training data inputs
            outputs_train (pandas.DataFrame): training data outputs
        """
        self._inputs_train = inputs_train
        self._outputs_train = outputs_train
        print("Training data stored")

    def predict(self, inputs):
        """
        Predicts an output given various inputs data
        Args:
            inputs (pandas.DataFrame): input data
        Returns:
            pandas.DataFrame: predicted outputs
        """
        inputs_arr = np.array(inputs)
        predicts = [self.predict_single(input) for input in inputs_arr] # List with various outputs
        return predicts

    def predict_single(self, input_test):
        """
        Function to predict the output given single input data row
        Args:
            input_test (pandas.DataFrame): input data, needs to have the Attributes of the training data
        Returns:
            pandas.DataFrame: predicted outputs
        """
        distances = []
        for i, input_row in enumerate(self._inputs_train):
            dist = self.euclidean_distance(input_row, input_test)   # measure distance between given inputs and local data
            distances.append((dist, self._outputs_train[i])) # save distance and output data

        sorted_distances = sorted(distances, key=lambda x: x[0])
        nearest_neighbors = sorted_distances[:self.k] # get only the distances up to k
        nearest_neighbor_label = [neighbor[1] for neighbor in nearest_neighbors]
        survived = Counter(nearest_neighbor_label).most_common(1)[0] # Check for most common output in the neighbors
        return survived[0]


    @staticmethod
    def euclidean_distance(p1, p2):
        return np.linalg.norm(p1 - p2)
