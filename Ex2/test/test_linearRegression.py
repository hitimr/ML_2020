# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import pytest
import numpy as np

from GD.LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from common import DataParser
import config as cfg


def test_LinearRegression():
    lg = LinearRegression()

    lg = LinearRegression(alpha=0.123)
    assert lg.alpha == 0.123

    lg = LinearRegression(max_iter=12345)
    assert lg.max_iter == 12345

    lg =  LinearRegression(weigths="residual")
    assert lg.weigth_mode == "residual"

    with pytest.raises(Exception): LinearRegression(weigths="wrong argument")



def test_fit():
    n_samples, n_features = 100, 10
    rng = np.random.RandomState(69)
    X = np.array([np.linspace(0,1, n_samples) for i in range(n_features)])
    y = np.array(10*X[0] + rng.rand(n_samples)*1)
    X = X.T

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = LinearRegression(alpha=0.001)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 score: {r2}")

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

def test_rss_vector():
    lg = LinearRegression()

    n = 10000
    x = np.random.rand(n)
    y = np.random.rand(n)
    w0 = np.random.rand()
    w1 = np.random.rand()

    # Reference calculation by 
    reference_sum = 0
    for i in range(len(x)):
        reference_sum += (y[i] - (w0 + w1*x[i]))**2

    lg._dot_yy = np.dot(y,y)
    lg._sum_y = np.sum(y)
    assert abs(reference_sum - lg.rss_vector(x,y,w0,w1)) < cfg.FP_TOLERANCE


def test_rss():

    def runParameters(n, m):
        lg = LinearRegression()
        X = np.random.rand(m,n)
        y = np.random.rand(n)
        w0 = np.random.rand(m)
        w1 = np.random.rand(m)

        reference_sums = []
        for j in range(m):
            sum = 0
            for i in range(n):
                sum += (y[i] - (w0[j] + w1[j]*X[j][i]))**2
            reference_sums.append(sum)
        reference_sums = np.array(reference_sums)

        rss_lg = lg.rss(X,y, w0, w1)
        assert np.allclose(reference_sums, rss_lg, cfg.FP_TOLERANCE)

    runParameters(10**4, 10)
    runParameters(10**4, 1)



if __name__ == "__main__":
    test_fit()