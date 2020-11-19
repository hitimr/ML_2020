import numpy as np

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

    def fit(self, x, y):
        """Fit the model

        Args:
            x_train (array-like): Training data of shape (n_samples, n_features)
            y_train (array-like): Target values
        """   
        x = np.array(x)
        y = np.array(y)     
        return

    def rss(self):
        return 0

    def initialize_w(self, x, y):
        w_0 = x.min()


if __name__ == "__main__":
    reg = LinearRegression()
    x, y = DataParser.parse_test_housePrices(splitData=True)

    reg.fit(x,y)