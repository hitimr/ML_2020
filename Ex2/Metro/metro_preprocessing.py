import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import log2


import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize

def preprocessing(df, transform = False):
    holiday_list = df.holiday.unique()
    for l in holiday_list:
        if l == "None":
            df = df.replace(l,0)
    for l in holiday_list:
        if l != "None":
            df = df.replace(l,1)

    indexNames = df.index[df.temp == np.min(df["temp"])]
    df.drop(indexNames , inplace=True)
    indexNames = df.index[df.rain_1h > 300]
    df.drop(indexNames , inplace=True)
    df["date_time"] = pd.to_datetime(df.date_time)
    df["hour"] = df.date_time.dt.hour
    df["day"] = df.date_time.dt.day
    df["month"] = df.date_time.dt.month
    df["year"] = df.date_time.dt.year

    df = df.drop("date_time", axis=1)

    lsitweather = df.weather_main.unique()
    l = 0
    for j in lsitweather:
        df = df.replace(j,l)
        l = l + 1

    lsitweather = df.weather_description.unique()
    l = 0
    for j in lsitweather:
        df = df.replace(j,l)
        l = l + 1

    if(transform):
        y = df["traffic_volume"]
        e = 0.21515151515151515
        y_mean = y.apply(lambda x: (x**e)).mean()
        y = y.apply(lambda x: (x**e)-y_mean)
    else:
        y = df["traffic_volume"]

    X = df[["temp","rain_1h","snow_1h","clouds_all","hour","day","month","year","weather_main","weather_description"]]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y
