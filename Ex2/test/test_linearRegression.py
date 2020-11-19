# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pytest
import numpy as np

from GD.LinearRegression import LinearRegression
from common import DataParser


def test_LinearRegression():
    lg = LinearRegression()


def test_check_Xy():
    lg = LinearRegression()
    X = np.array([[1,2,3],[4,5,6]])
    y = np.array([1,2,3])
    lg.check_Xy(X, y)

    with pytest.raises(Exception): lg.check_Xy(X,X) # Y is 2D
    with pytest.raises(Exception): lg.check_Xy(Y,Y) # X is 1D
    with pytest.raises(Exception): lg.check_Xy(Y,X) # arguments flipped     
    with pytest.raises(Exception): lg.check_Xy(X,[]) # dimensions of X and y font match

def test_initialize_w():
    lg = LinearRegression()

    # Test for dim(X) = 1
    X = np.array([[0,1,2,3]])
    y = np.array([0, 2, 4, 6])
    w0, w1 = lg.initialize_w(X,y)
    assert w0[0] == 0
    assert w1[0] == 2

    # Test for dim(X) = 2
    X1 = [1,2,5,4]
    X2 = [2, 4, 32, 16]
    X = np.array([X1, X2])
    y = np.array([2, 3, 4, 5])

    w0, w1 = lg.initialize_w(X,y)
    for i in range(len(w0)):
        assert 4 == w0[i]+w1[i]*X[i][-2]




if __name__ == "__main__":
    test_initialize_w()