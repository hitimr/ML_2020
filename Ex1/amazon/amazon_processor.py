import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from math import log2

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
import time
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier as RFC


from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pathlib
import os
import sys

def FeatureSelection_Varianz(per):
    return VarianceThreshold(threshold=(per * (1 - per)))

def FeatureSelection_kBest(k):
    return SelectKBest(chi2, k=k)

def FeatureSelection_PCA(k):
    return PCA(n_components=k)


def Preprocessing_Amazon(df_train, df_target, feature_method = "kBest", k = 50, per = 0.8, scale_method = "standard"):
    if(feature_method == "varianz"):
        fea = FeatureSelection_Varianz(per) 
        df_train = fea.fit_transform(df_train)
    if(feature_method == "kBest"):
        fea = FeatureSelection_kBest(k)
        df_train = fea.fit_transform(df_train, df_target)
    if(feature_method == "PCA"):
        fea = FeatureSelection_PCA(k)
        df_train = fea.fit(df_train.values).transform(df_train.values)


    if(scale_method == "standard"):
        scl = StandardScaler() 
    if(scale_method == "minmax"):
        scl = MinMaxScaler()
    if(scale_method == "quantil"):
        scl = QuantileTransformer()
    
    scl.fit(df_train)
    df_train = scl.transform(df_train)

    return pd.DataFrame(df_train)