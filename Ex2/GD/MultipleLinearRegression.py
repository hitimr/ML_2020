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




class MultipleLinearRegression():
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
    w0 = []
    w1 = []

    def __init__(self, metric="RSS", alpha=0.0001, max_iter=1000, weigths="uniform"):
        if metric == "RSS":
            self.metric = self.rss_vector

        self.alpha = alpha

        assert max_iter > 0 # Error: Invalid argument for max_iter
        self.max_iter = max_iter
        self.w0 = []
        self.w1 = []
        
        assert weigths == "uniform" or weigths == "residual"    # Error invalid argument for weigths
        self.weigth_mode = weigths

    def fit(self, X, y):
        """Fit the model

        Args: X (array-like): Training data of shape (n_samples, n_features) y
            (array-like): Target values
        """
        X, y = self.sanitizeInputXy(X, y)
        
        w0, w1 = self.initialize_w(X, y)
        self.y = y
        for i in range(len(X)):
            w0_i, w1_i = self.gradientDescend(X[i], w0[i], w1[i])
            if np.isnan(w0_i) or np.isnan(w0_i):
                raise SystemError("Gradient is diverging! try to use smaller alpha")

            w0[i] = w0_i
            w1[i] = w1_i

        self.w0 = w0
        self.w1 = w1       

        if self.weigth_mode != "uniform":   
            weigths = 1.0 / self.rss(X, y, w0, w1)      
            self.weigths = weigths / np.sqrt(np.sum(weigths**2)) # Normalize weigths

        else:
            self.weigths = np.ones(len(w0), dtype=float)

        return self

    def fit2(self, X, y):
        X, y = self.sanitizeInputXy(X, y)
        
        w0, w1 = self.initialize_w(X, y)


        
    def gradientDescend(self, x, w0, w1):
        iter_cnt = 0

        n = len(x)
        dot_xx = np.dot(x,x)
        dot_xy = np.dot(x,self.y)
        sum_x = sum(x)
        sum_y = sum(self.y)

        # Iteration for Gradient descend
        while(True):  
            iter_cnt += 1     
            
            grad_w0 = 2*self.alpha*(n*w0 + w1*sum_x - sum_y)
            grad_w1 = 2*self.alpha*(w1*dot_xx + w0*sum_x - dot_xy)
            w0 = w0 - grad_w0
            w1 = w1 - grad_w1

            # stop if we reach the limit of iterations
            if iter_cnt > self.max_iter:
                return w0, w1         

    def gradientDescend_matrix(self, args):
        # Experimantal implementation by solving everything using matrix ops
        X, w0, w1 = args[0], args[1], args[2]
        iter_cnt = 0
        n = len(X[0])
        dot_xx = np.matmul(X.T,X)
        dot_xy = np.dot(X,y)
        sum_x = np.array([sum(x) for x in X])
        sum_y = sum(self.y)

        while(True):  
            iter_cnt += 1     
            
            grad_w0 = 2*self.alpha*(n*w0 + w1*sum_x - sum_y)
            grad_w1 = 2*self.alpha*(np.dot(dot_xx, w1) + np.dot(sum_x, w0) - dot_xy)
            w0 = w0 - grad_w0
            w1 = w1 - grad_w1

            # stop if we reach the limit of iterations
            if iter_cnt > self.max_iter:
                return w0, w1             



    def predict(self, X):
        # TODO add functionality for single sample
        if len(self.w0) == 0:
            raise(SystemError("Model is not fitted"))

        if (len(self.w0) != len(X[0])) or (len(self.w1) != len(X[0])):
            raise(ValueError("Dimensions of X does not match w0 and w1"))

        y = []
        for x in X:
            y.append( np.average(self.w0 + x*self.w1, weights=self.weigths))
        return np.array(y)

    def rss_vector(self, x, y, w0, w1):
        """Calculate residual sum of squares for x being an array of size n.
        Original algorithm from lecture slides which was then optimized for
        performance The implemented algorithm is the one with the best runtime
        from bechmark_rss_vector.py

        Args: x (np.array): x values y (np.array): y values w0 (float): offset
            w1 (float): slope

        Returns: float: residual
        """        
        return self._dot_yy + w1*w1*np.dot(x,x) - 2*w1*np.dot(x,y) + len(x)*w0*w0 \
        - 2*w0*self._sum_y + 2*w0*w1*sum(x)

    def rss(self, X, y, w0, w1):
        """Calculate the residual sum of squares for X being a Matrix of size 
        n x m

        Returns: np.array: vector vectorcontaining m residua
        """  
        self._dot_yy = np.dot(y,y)
        self._sum_y = np.sum(y)
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

        w0, w1 = [], []
        for x in X:
            i_0 = x.argmin() 
            i_1 = x.argmax()

            # calculate slope and offset
            k = (y[i_1] - y[i_0]) / (x[i_1] - x[i_0])
            d = y[i_0] - k*x[i_0]

            w1.append(k)
            w0.append(d)

        
        return np.array(w0), np.array(w1)

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
            
            self.check_Xy(X.T,y)
            return X.T,y


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
    alpha = 0.0001

    
    n_samples, n_features = 5000, 10
    noise = 0.15
    rng = np.random.RandomState(0)
    X = np.array([np.linspace(0,1, n_samples) for i in range(n_features)])

    y = np.zeros(n_samples)
    for i in range(n_features):
        y += X[i]*(i+1)
    y = y * (1 + rng.rand(n_samples)*noise) # add noise

    X = X.T

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1 )


    my_reg = MultipleLinearRegression(alpha=alpha, weigths="residual")

    start_time = time()
    my_reg.fit(X_train, y_train)
    end_time = time()
    time_my = end_time - start_time


    
    sk_reg = linear_model.SGDRegressor(alpha=alpha)
    start_time = time()
    sk_reg.fit(X_train,y_train)
    end_time = time()
    time_sk = end_time - start_time

    y_pred_my = my_reg.predict(X_test)
    y_pred_sk = sk_reg.predict(X_test)


    print(f"R2 SK Learn: {r2_score(y_test, y_pred_sk)}, time = {time_sk}")
    print(f"R2  ML 2020: {r2_score(y_test, y_pred_my)}, time = {time_my}" )
    


