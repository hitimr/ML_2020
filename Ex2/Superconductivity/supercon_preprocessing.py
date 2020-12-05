import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import log2

import matplotlib as mpl

from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize


def preprocessing(df,transform = None):

    if (transform):
        y = df['critical_temp']
        e=0.1515151515151516
        y_mean = y.mean()
        y = y.apply(lambda x: (x**e)-y_mean)
    else:
        y = df['critical_temp']
    X = np.array(df.drop("critical_temp", axis = 1))

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y

