import numpy as np
import pandas as pd
import numpy as np

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
        self.k = k

    def store(self, inputs_train, outputs_train):
        """
        Stores training data
        Args:
            inputs_train (pandas.DataFrame): training data inputs
            outputs_train (pandas.DataFrame): training data outputs
        """
        self.x_train = inputs_train
        self.y_train = outputs_train
        print("Training data stored")

    def search(self, df: pd.DataFrame):
        cols = df.columns.tolist()

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))