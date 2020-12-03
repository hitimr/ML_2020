import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import log2


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize

def preprocessing(df, drop_missing_values = None):
    cols_to_drop = ['Team', 'League', 'Year', 'RankSeason', 'RankPlayoffs', 'Playoffs']
    df = df.drop(cols_to_drop, axis=1)
    df[['OOBP','OSLG']] = df[['OOBP','OSLG']].astype(float)

    if(drop_missing_values):
        df = df.dropna()
    else:
        r_mean = np.mean(df["OOBP"])
        df["OOBP"] = df["OOBP"].replace(float('NaN'),r_mean)
        r_mean = np.mean(df["OSLG"])
        df = df.replace(float('NaN'),r_mean)

    X = df[["RS","RA","OBP","SLG","BA","G","OOBP","OSLG"]]
    Y = df["W"]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, Y





    
