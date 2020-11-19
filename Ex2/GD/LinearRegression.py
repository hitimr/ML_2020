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
    """
    Custom Classifier for linear Regression.

    The basic structure and methods used are similar to the models provided by
    sklearn to ensure compatibility for our model trainer. The actual logic is
    implemented from scratch except baisc operations such as matrix/vector
    operations   

    fit() will try to work with as many types of arguments as possible (list,
    DataFrame, arrays, etc) and tries to convert them accordingly. All
    subsequent calculations require numpy data-structures

    A note about naming convention: n -> number of samples m -> number of
        features

        x -> vector (size = n)
        X -> matrix (size = n x m)

    """
    def __init__(self, metric="RSS", alpha=0.001):
        if metric == "RSS":
            self.costFunction = self.rss

        self.alpha = alpha

    def fit(self, X, y):
        """Fit the model

        Args: X (array-like): Training data of shape (n_samples, n_features) y
            (array-like): Target values
        """   
        X = X.transpose()
        y = np.array(y).flatten()

        self.initialize_w(X,y)   
        return

    def iterate(self, X, y, w0, w1):

        rss = self.rss(X, y, w0, w1)
        grad_w0 = self.alpha*np.gradient(rss, w0)
        grad_w1 = self.alpha*np.gradient(rss, w1)
        
        w0 = w0 - grad_w0
        w1 = w1 - grad_w1

        return w0, w1


    def rss_vector(self, x, y, w0, w1):
        """Calculate residual sum of squares for x being an array of size n.
        Original algorithm from lecture slides which was then optimized for
        performance The implemented algorithm is the one with the best runtime
        from bechmark_rss_vector.py

        TODO: pull out dot-products as they dont change over iterations

        Args: x (np.array): x values y (np.array): y values w0 (float): offset
            w1 (float): slope

        Returns: float: residual
        """        
        return np.dot(y,y) + w1*w1*np.dot(x,x) - 2*w1*np.dot(x,y) + len(x)*w0*w0 \
        - 2*w0*np.sum(y) + 2*w0*w1*sum(x)

    def rss(self, X, y, w0, w1):
        """Calculate the residual sum of squares for X being a Matrix of size 
        n x m

        Args: X (np.matrix): Matrix of x values y (np.array): y values w0
            (np.array): offsets (size = m) w1 (np.array): slopes  (size = m)

        Returns: np.array: vector vectorcontaining m residua
        """        
        return np.asarray([
            self.rss_vector(X[j], y, w0[j], w1[j]) 
            for j in range(len(X))
            ])
        

    def initialize_w(self, X, y):
        """Initialize w0 (offset) and w1 (slope). Initial values are chosen by
        finding the smallest and largest x (x_min, x_max) Then we do linear
        interpolation for f(x_min) and f(x_max)

        Args: X (np.matrix): Matrix of x values y (np.array): y values

        Returns: np.array, np.array: w0 (list of offets), w1 (list of
            slopesslope)
        """        
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

        return np.array(w_0), np.array(w_1)


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
        assert isinstance(X, np.ndarray)    # wrong instance
        assert isinstance(y, np.ndarray)    # wrong instance

        assert len(y.shape) == 1    # y must be 1D
        assert len(X.shape) == 2    # X must be 2D
        assert len(X[0]) == len(y)  # dimensions must match



if __name__ == "__main__":
    reg = LinearRegression(alpha=0.001)
    X, y = DataParser.parse_test_housePrices(splitData=True)
    X = X.to_numpy().transpose()
    y = y.to_numpy().flatten()
    x_vals = list(range(1,5))
    plt.scatter(X[0], y)

    w0, w1 = reg.initialize_w(X, y) 


    plt.plot(x_vals, [w0[0] + w1[0]*x for x in x_vals])

    for i in range(10):
        w0, w1 = reg.iterate(X, y, w0, w1)
        #plt.plot(x_vals, [w0[0] + w1[0]*x for x in x_vals])

    #plt.show()
