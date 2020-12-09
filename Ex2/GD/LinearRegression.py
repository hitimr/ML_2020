import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import r2_score

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import config
from common import DataParser

class LinearRegression():
    def __init__(self, alpha=0.0001, max_iter=1000, init_mode="linear"):
        """This class useds gradient descent to find the best linear
        approximation for a givenb dataset. The class is structures similar to
        the models of SK Learn to ensure compatibility

        Example Usage:
            model = LinearRegression()  
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)


        Args: 
            alpha (float, optional): learning rate.

            max_iter (int, optional): maximum number of iterations. 

            init_mode (str, optional): "linear" - use linear interpolation
                to initialize weights; "constant": use 0 to initialize weigths.
                Defaults to "linear".
        """

        assert max_iter > 0
        assert alpha > 0

        self.alpha = alpha        
        self.max_iter = max_iter
        self.c = 0
        self.w = np.array([])
        self.init_mode = init_mode


    def fit(self, X, y):
        """Fit the model using grtadient descent. See slides for the algorimth explanation

        Args:
            X (np.ndarray): 2D array containing training data
            y (np.array): 1D array containing training target

        """
        X = np.array(X)
        y = np.array(y)

        c, w = self.initial_w(X, y) # Initialize c, w

        self.y = y
        n = float(len(X[0]))
        X = X.T
        iter_cnt = 0

        # Iterrate
        while(True):
            residual = y - np.dot(w, X) - c

            # calculate derivatives
            grad_w =  - 2.0 / n * np.dot(X, residual)
            grad_c =  - 2.0 / n * np.sum(residual)

            # update simulatinously
            w = w - self.alpha * grad_w          
            c = c - self.alpha * grad_c

            # stop if we have reached the maximum amount of iterarations
            iter_cnt += 1
            if iter_cnt >= self.max_iter:
                break

        self.c = c
        self.w = w

        return self

    def initial_w(self, X, y):
        self.check_Xy(X, y)

        # Initialize with linear interpolation. See slides for more information
        if self.init_mode == "linear":
            c, w = [], []

            for x in X.T:
                i_0 = x.argmin()
                i_1 = x.argmax()

                # calculate slope
                k = (y[i_1] - y[i_0]) / (x[i_1] - x[i_0])
                d = y[i_0] - k*x[i_0]

                w.append(k)
                c.append(d)
            c = np.average(c)

        # initialize all c, w_i with 0
        elif self.init_mode == "constant":
            c = 0
            w = np.zeros(len(X[0]), float)

        else:
            raise(ValueError("Unknown argument for init_mode"))

        return c, w

    def sanitizeInputXy(self, X, y):
        # check if input is works. If so just return it
        try:
            self.check_Xy(X,y)
            return X, y

        except:
            # Something is not algrigh with the input. Lets try to fix it
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy().transpose()

            else:
                X = np.array(X)
            if isinstance(y, pd.DataFrame):
                y = y.to_numpy().flatten()
            else:
                y = np.array(y)
            
            self.check_Xy(X,y)
            return X ,y

            
    def predict(self, X):
        X = np.array(X)

        if len(self.w) == 0:
            raise(SystemError("Model is not fitted"))

        if (len(self.w) != len(X[0])):
            raise(ValueError("Dimensions of X does not match w1"))

        # Calculate c + w0 * x0 + w1 * x1 + w2 * x2
        y_pred = np.dot(self.w, X.T) + self.c
        return y_pred


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
        assert len(X) == len(y)  # dimensions must match


# For testing and debugging
if __name__ == "__main__":
    alpha = 0.001
    max_iter = 5000

    
    m_samples, n_features = 2000, 10
    noise = 0.15
    rng = np.random.RandomState(0)

    X = rng.rand(m_samples, n_features)       
    w = rng.rand(n_features)
    c = 2

    y = np.dot(X, w) + c


    #X, y = DataParser.parse_test_housePrices(splitData=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1 )


    my_reg = LinearRegression(alpha=alpha, max_iter=max_iter)

    start_time = time()
    my_reg.fit(X_train, y_train)
    end_time = time()
    time_my = end_time - start_time


    
    sk_reg = linear_model.SGDRegressor(alpha=alpha, max_iter=max_iter)
    start_time = time()
    sk_reg.fit(X_train, y_train)
    end_time = time()
    time_sk = end_time - start_time

    y_pred_my = my_reg.predict(X_test)
    y_pred_sk = sk_reg.predict(X_test)


    print(f"R2 SK Learn: {r2_score(y_test, y_pred_sk)}, time = {time_sk}")
    print(f"R2  ML 2020: {r2_score(y_test, y_pred_my)}, time = {time_my}" )