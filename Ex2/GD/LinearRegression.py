import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import config
from common import DataParser




class LinearRegression():

    def __init__(self, metric="RSS"):
        if metric == "RSS":
            self.costFunction = self.rss

    def fit(self, X, y):
        """Fit the model

        Args:
            X (array-like): Training data of shape (n_samples, n_features)
            y (array-like): Target values
        """   
        X = X.transpose()
        y = np.array(y).flatten()

        self.initialize_w(X,y)   
        return

    def check_Xy(self, X, y):
        """Check if X and y are suitable arguments for further calculations
        X must be a Matrix
        y must be a vector

        Args:
            X (): [description]
            y (): [description]    

        Raises:
            AssertionError: If any check fails
        """        
        assert isinstance(X, np.ndarray)    # wrong instance
        assert isinstance(y, np.ndarray)    # wrong instance

        assert len(y.shape) == 1    # y must be 1D
        assert len(X.shape) == 2    # X must be 2D
        assert len(X[0]) == len(y)  # dimensions dont match
        return True


    def rss(self):
        return 0

    def initialize_w(self, X, y):
        self.check_Xy(X, y)

        w_0, w_1 = [], []
        for x in X:
            i_0 = x.argmin() 
            i_1 = x.argmax()

            # calculate slope and offset
            k = (y[i_1] - y[i_0]) / (x[i_1] - x[i_0])
            d = y[i_0] - k*x[i_0]

            w_1.append(k)
            w_0.append(d)

        return w_0, w_1


if __name__ == "__main__":
    reg = LinearRegression()
    X, y = DataParser.parse_test_housePrices(splitData=True)
    X = X.to_numpy().transpose()
    y = y.to_numpy().flatten()

    w0, w1 = reg.initialize_w(X, y)
    
    x_vals = list(range(1,5))
    y_vals = [w0[0] + w1[0]*x for x in x_vals]
    plt.scatter(X[0], y)
    plt.plot(x_vals, y_vals)
    plt.show()