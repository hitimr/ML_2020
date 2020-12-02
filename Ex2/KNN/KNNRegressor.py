import inspect
import os
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import config
from common import DataParser
from common.runtime import runtime

class KNNRegressor():
    """
    Custom KNN Regressor implementation.

    Reference:
    (1): https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
    (2): https://towardsdatascience.com/k-nearest-neighbors-classification-from-scratch-with-numpy-cb222ecfeac1
    (3): https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/neighbors/_regression.py#L160

    Basic flow:
    1. Load the data
    2. Initialise the value of k=n_neighbors
    3. For getting the predicted class, iterate from 1 to total number of training data points
        3.1. Calculate the distance between test data and each row of training data. Here we will use Euclidean distance as our distance metric since it’s the most popular method. The other metrics that can be used are Chebyshev, cosine, etc.
        3.2. Sort the calculated distances in ascending order based on distance values
        3.3. Get top k rows from the sorted array
        3.4. Get the most frequent class of these rows
        3.5. Return the predicted class

    Numpy functions used:
        - np.mean
        - np.partition
        - np.linalg.norm
        - np.take_along_axis: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html#numpy.take_along_axis

    """

    #@runtime
    def __init__(self,
                 n_neighbors: int = 5,
                 p: int = 2,
                 weights: str = "uniform",
                 params_dict: dict = None,
                 dist_func=np.linalg.norm,
                 debug: bool = False):
        self._training_x = []
        self._training_y = []
        self._num_samples = 0
        self._params = {"n_neighbors": n_neighbors}
        if params_dict:
            self._params = params_dict
        else:
            self._params["n_neighbors"] = n_neighbors
            self._params["p"] = p
            self._params["weights"] = weights

        self._dist_func = dist_func
        self.KNN_DEBUG = debug

    @property
    def training_data(self):
        return self._training_x, self._training_y

    @training_data.getter
    def training_y(self):
        return self._training_y

    @training_data.getter
    def training_x(self):
        return self._training_x

    @training_data.setter
    def training_data(self, data):
        if isinstance(data, tuple):
            self._training_x = data[0], self._training_y = data[1]
            self._num_samples = self._training_x.shape[0]
        elif isinstance(data, dict):
            data = {}
            assert "X" in data.keys() and "y" in data.keys()
            self._training_x = data["X"], self._training_y = data["y"]
            self._num_samples = self._training_x.shape[0]

    @training_data.setter
    def training_data(self, X, y):
        self._training_x = X, self._training_y = y
        self._num_samples = self._training_x.shape[0]

    def get_params(self):
        return self._params

    #@runtime
    def fit(self, X, y):
        """Fit the model

        Args: 

        Really just stores the data...
        """
        X, y = self.sanitizeInputXy(
            X, y)  # need to actually rebind X, y here :D --> use return values
        self._training_x = X
        self._training_y = y
        self._num_samples = len(y)

        if self._num_samples < self._params["n_neighbors"]:
            print(
                "n_neighbors was too large - n_neighbors > num_samples ==> resized n_neighbors = num_samples"
            )
            self._params["n_neighbors"] = self._num_samples
        return self

    def sanitizeInputXy(self, X, y):
        # check if input is works. If so just return it
        try:
            self.check_Xy(X, y)
            return X, y

        except:

            print("Trying to fix...")
            # Something is not algrigh with the input. Lets try to fix it
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            else:
                X = np.array(X)

            if isinstance(y, pd.DataFrame):
                y = y.to_numpy()
            else:
                y = np.array(y)

            if len(y.shape) > 1:
                y = y.ravel()

            self.check_Xy(X, y)
            return X, y

    def check_Xy(self, X, y):
        """ Convenience function.
        Check if X and y are suitable arguments for further calculations
        X must be a Matrix of size n x m
        y must be a vector of size n

        Args:
            X (): [description]
            y (): [description]    

        Raises:
            AssertionError: If any check fails
        """
        assert isinstance(X, np.ndarray)  # wrong instance
        assert isinstance(y, np.ndarray)  # wrong instance
        if len(y.shape) > 1:
            mask = [True if x==1 else False for x in y.shape]
            idx = mask.index(False)
            assert any(mask), str(y.shape) + str(mask)  # y must be 1D
            y_len = y.shape[idx]
        else:
            y_len = len(y)
        assert len(X.shape) == 2, str(X.shape)  # X must be 2D
        assert X.shape[0] == y_len, (X.shape[0], y_len)  # dimensions must match

    #@runtime
    def predict(self, X, ret_distances=0):
        # TODO add functionality for single sample
        if len(self._training_x) == 0 or len(self._training_y) == 0:
            raise (SystemError("Model is not fitted"))
        if len(self._training_x) != len(self._training_y):
            raise (SystemError("X,y - training data not equally long!"))

        if len(X.shape) > 1:
            if X.shape[1] == 1:
                # single instances
                return self._predict_single(X, ret_distances)

            # multiple instances
            return self._predict_multi(X, ret_distances)

        # single instances
        return self._predict_single(X, ret_distances)

    def k_nearest(self, X, ret_distances=0):
        n_neighbors = self._params["n_neighbors"]
        if self.KNN_DEBUG:
            print("args")
            print(X)
            print(ret_distances)
            print("params")
            print(self._params)

        # should be at least 2D np.arrays --> so matrices
        if isinstance(self._training_x, pd.DataFrame):
            diff = self._training_x.to_numpy()
        else:
            diff = self._training_x
        if isinstance(X, pd.DataFrame):
            diff = diff - X.to_numpy()[:, np.newaxis, :]
        else:
            diff = diff - X[:, np.newaxis, :]

        distances = self._dist_func(diff, ord=self._params["p"], axis=2)
        neighbors = np.argpartition(distances, n_neighbors)[:,:n_neighbors]
        if self.KNN_DEBUG:
            print("diff:")
            print(diff)
            print("distances")
            print(distances)
            print("neighbors")
            print(neighbors.dtype)
            print(neighbors)
        if ret_distances == 1:
            return neighbors, distances
        if ret_distances == 2:
            return neighbors, distances[neighbors]
        return neighbors

#### PRIVATES #####

    def _predict_single(self, X, ret_distances):
        print("Single instance")
        return self._predict_multi(X, ret_distances)

    def _predict_multi(self, X, ret_distances):
        if ret_distances:
            neighbors, distances = self.k_nearest(X, ret_distances=1)
        else:
            neighbors = self.k_nearest(X)

        if self._params["weights"] is None or self._params["weights"] is "uniform":
            if self.KNN_DEBUG:
                print(self._params["weights"])
            print("neighbors")
            print(neighbors.dtype)
            print(neighbors)
            # y = np.mean(sorted[:, :n_neighbors], axis=1)
            axis = 0
            print(f"subset (taken from axis={axis}):")
            subset = []
            for sample_neighbors in neighbors:
                subset.append(np.take_along_axis(self._training_y, sample_neighbors, axis=axis))
            subset = np.array(subset)
            print(subset)
            y = np.mean(subset, axis=1)

        if ret_distances:
            return y, distances
        return y

    def _get_weights(self, weight_type, arg):
        if weight_type == "uniform":
            return np.array([1/n_neighbors for n_neighbors in range(1, arg+1)])
        if weight_type == "distance":
            with np.errstate(divide='ignore'):
                weights = 1. / arg
            inf_mask = np.isinf(weights)
            inf_row = np.any(inf_mask, axis=1)
            weights[inf_row] = inf_mask[inf_row]
            return weights
        else:
            raise(ValueError("Weights can be either uniform, distance"))

#### MAGIC ####

    def __repr__(self):
        return repr('KNNRegressor: ' + str(self._params))

    def __str__(self):
        n_neighbors = self._params["n_neighbors"]
        return f'KNNRegressor with n_neighbors={n_neighbors}'


######
# if __name__ == "__main__":
#     print("Here")
