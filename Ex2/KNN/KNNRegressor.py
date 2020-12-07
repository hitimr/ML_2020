# std
import inspect
import os
import sys
# packgaes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# std
from time import time
from math import floor
# packages
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# modules : setup
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# modules : import
import config
from common import DataParser
from common.runtime import runtime

# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function


####
class ChunkViewer:
    """ Custom class to prodive an iterable chunked view of a numpy ndarray.
        Can help limit the amount of memory needed concurrently.

        Usage: see KNNRegressor
    """
    def __init__(self, chunk_size, array_size):
        self.array_size = array_size
        self.chunk_size = chunk_size

        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        """Iterator for the index range [start, stop).

        Returns: start, stop
        """
        # will be [start, stop) as is customary
        start = self.current
        stop = self.array_size if (
            start + self.chunk_size) > self.array_size else (start +
                                                             self.chunk_size)
        self.current = stop
        #print(start, stop)
        if start != stop:
            return start, stop
        raise StopIteration

    def generator(self, arr: np.ndarray = None):
        """Generator that yields either
            (i) an iterated view of a numpy array, if arr is numpy array,
            (ii) or index ranges as above, if arr is not a numpy array or not used.
        """
        if isinstance(arr, np.ndarray):
            while True:
                try:
                    start, stop = self.__next__()
                    yield arr[start:stop, ...]
                except StopIteration:
                    break
        else:
            while True:
                try:
                    yield self.__next__()
                except StopIteration:
                    break

    def next_as_view(self, arr: np.ndarray):
        # will be [start, stop) as is customary
        start, stop = next(self)
        return arr[start:stop, ...]


#######


class KNNRegressor():
    """
    Our custom KNN Regressor implementation.
    Model parameters are named the same as in sklearn for ease of use and understanding.

    Args:
        n_neighbors...  number of nearest neighbors (often called k),
        p...            order of norm,
        weights...      does nothing at the moment as we did not 
                        finish implementing weighted average,
        chunk_size...   sets size of chunks of prediction samples for which
                        the nearest neighbors are searched for,
        params_dict...  entry point for setting model parameters via
                        a dictionary (same as for sklearn),
        dist_func...    the distance function used,
        dist_params...  arguments passed to distance functions (overrides p),
        profile...      switch to turn on profiling mode,
        debug...        switch to turn on debug output

    

    Numpy functions used:
        - np.mean
        - np.argpartition
        - np.linalg.norm
        - np.take_along_axis: https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html#numpy.take_along_axis

    """

    #@runtime
    def __init__(self,
                 n_neighbors: int = 5,
                 p: int = 2,
                 weights: str = "uniform",
                 chunk_size: int = 4,
                 params_dict: dict = None,
                 dist_func = np.linalg.norm,
                 dist_params: dict = None,
                 profile: bool = False,
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
        self._params["algorithm"] = "brute"

        self._chunk_size: int = chunk_size

        self._dist_func = np.max
        if p:
            self._dist_func = dist_func
            self._dist_params = {"ord": p, "axis": 2}
        if dist_params == None:
            self._dist_params = {"axis": 2}
        else:
            self._dist_params = dist_params

        self.PROFILE = profile
        self.profile = {
            "argpartition": [],
            "distances": [],
            "mean": [],
            "neighbors": [],
            "subset": []
        }
        self.KNN_DEBUG = debug

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, size: int):
        if size < 1:
            print("Invalid chunk size: rejected!")
            #return self._chunk_size
        self._chunk_size = size

    # return self._chunk_size

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

        Really just stores the data as numpy arrays and checks if dimensions of data align.
        """
        self._training_x, self._training_y = self.sanitizeInputXy(X, y)
        self._num_samples = len(self._training_y)

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
            #print("Trying to fix...")
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

        Raises:
            AssertionError: If any check fails
        """
        assert isinstance(X, np.ndarray)  # wrong instance
        assert isinstance(y, np.ndarray)  # wrong instance
        if len(y.shape) > 1:
            mask = [True if x == 1 else False for x in y.shape]
            idx = mask.index(False)
            assert any(mask), str(y.shape) + str(mask)  # y must be 1D
            y_len = y.shape[idx]
        else:
            y_len = len(y)
        assert len(X.shape) == 2, str(X.shape)  # X must be 2D
        assert X.shape[0] == y_len, (X.shape[0], y_len
                                     )  # dimensions must match

    #@runtime
    def predict(self, X, ret_distances=0):
        """Predict on the samples X.

        Args:
            X... data
            ret_distances...    =1 -> returns distances
                                =2 -> returns k smallest distances
                                mostly useful for debugging
        """
        if len(self._training_x) == 0 or len(self._training_y) == 0:
            raise (SystemError("Model is not fitted"))
        if len(self._training_x) != len(self._training_y):
            raise (SystemError("X,y - training data not equally long!"))

        if len(X.shape) > 1:
            if X.shape[1] == 1:
                # single instances
                if isinstance(X, pd.DataFrame):
                    return self._predict_single(X.to_numpy(), ret_distances)
                else:  # assume its numpy array
                    return self._predict_single(X, ret_distances)

            # multiple instances
            if isinstance(X, pd.DataFrame):
                return self._predict_multi(X.to_numpy(), ret_distances)
            else:  # assume its numpy array
                return self._predict_multi(X, ret_distances)

        # single instances
        if isinstance(X, pd.DataFrame):
            return self._predict_single(X.to_numpy(), ret_distances)
        else:  # assume its numpy array
            return self._predict_single(X, ret_distances)

    def k_nearest(self, X, ret_distances=0):
        """Compute k nearest neighbors.

            Returns:
                array of indices of neighbors for each sample X, 
                optional: [distances]
        """
        n_neighbors = self._params["n_neighbors"]
        if self.KNN_DEBUG:
            print("args")
            print(X)
            print(ret_distances)
            print("params")
            print(self._params)
        if self.PROFILE:
            start = time()
        # distances = self._dist_func(self._training_x - X[:, np.newaxis, :],
        #                             ord=self._params["p"],
        #                             axis=2)
        distances = self._dist_func(self._training_x - X[:, np.newaxis, :],
                                    **self._dist_params)
        if self.PROFILE:
            self.profile["distances"].append(time() - start)
            start = time()
        neighbors = np.argpartition(distances, n_neighbors)[:, :n_neighbors]
        if self.PROFILE:
            self.profile["argpartition"].append(time() - start)
        if self.KNN_DEBUG:
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
        return self._predict_multi(X, ret_distances, samples=1)

    def _predict_multi(self, X, ret_distances, samples: int = 0):
        if samples:
            prediction_size = samples
        else:
            prediction_size = X.shape[0]
        if self.KNN_DEBUG:
            print(f"number of samples to predict for : {prediction_size}")

        if prediction_size > self._chunk_size:
            #print("chunking...")
            chunk_size = self._chunk_size
            viewer = ChunkViewer(chunk_size, prediction_size)
            if self.PROFILE:
                start = time()
            neighbors = np.concatenate(
                list(map(self.k_nearest, viewer.generator(X))))
            if self.PROFILE:
                self.profile["neighbors"].append(time() - start)
            if self.KNN_DEBUG:
                print(f"self._chunk_size > prediction_size")
                print(f"{self._chunk_size} > {prediction_size}")
                print(
                    f"chunks in samples: {1+floor(samples/self._chunk_size)}")
                print("returning neighbors:")
                print(neighbors)

        else:
            if self.PROFILE:
                start = time()
            if ret_distances:
                neighbors, distances = self.k_nearest(X, ret_distances=1)
            else:
                neighbors = self.k_nearest(X)
            if self.PROFILE:
                self.profile["neighbors"].append(time() - start)

        # we ignore the weights for now
        if self.KNN_DEBUG:
            print(self._params["weights"])
            print("neighbors")
            print(neighbors.dtype)
            print(neighbors)
        # y = np.mean(sorted[:, :n_neighbors], axis=1)
        axis = 0
        if self.PROFILE:
            start = time()
        subset = np.array([
            np.take_along_axis(self._training_y, sample_neighbors, axis=axis)
            for sample_neighbors in neighbors
        ])
        if self.PROFILE:
            self.profile["subset"].append(time() - start)

        if self.KNN_DEBUG:
            print(f"subset (taken from axis={axis}):")
            print(subset)
        if self.PROFILE:
            start = time()
        y = np.mean(subset, axis=1)
        if self.PROFILE:
            self.profile["mean"].append(time() - start)

        if ret_distances == 1:
            return y, distances
        if ret_distances == 2:
            return y, neighbors
        return y
#####

    def _get_weights(self, weight_type, arg):
        if weight_type == "uniform":
            return np.array(
                [1 / n_neighbors for n_neighbors in range(1, arg + 1)])
        if weight_type == "distance":
            with np.errstate(divide='ignore'):
                weights = 1. / arg
            inf_mask = np.isinf(weights)
            inf_row = np.any(inf_mask, axis=1)
            weights[inf_row] = inf_mask[inf_row]
            return weights
        else:
            raise (ValueError("Weights can be either uniform, distance"))

#### MAGIC ####

    def __repr__(self):
        return repr('KNNRegressor: ' + str(self._params) +
                    f', chunk_size={self._chunk_size}')

    def __str__(self):
        n_neighbors = self._params["n_neighbors"]
        return f'KNNRegressor with n_neighbors={n_neighbors}, chunk_size={self._chunk_size}'


######
# if __name__ == "__main__":
#     print("Here")
