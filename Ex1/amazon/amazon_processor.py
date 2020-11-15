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

def FeatureSelection_kBest(k):
    return SelectKBest(chi2, k=k)

def FeatureSelection_PCA(k):
    return PCA(n_components=k)


def Preprocessing_Amazon(df_train, df_target, feature_method = "kBest", k = 50, per = 0.8, scale_method = "standard"):
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


def PlotHist(df,ylabel,xlabel,title,savename,bins,size):
    mpl.style.use('seaborn')
    plt.rc('xtick', labelsize=size) 
    plt.rc('ytick', labelsize=size) 
    fig = plt.figure(figsize=(40,12))
    plt.grid()
    plt.ylabel(ylabel, fontsize=size)
    plt.xlabel(xlabel, fontsize=size)
    plt.xlabel(xlabel, fontsize=size)
    #plt.title(title, fontsize=size)
    plt.hist(df["Class"],bins,facecolor='g')
    plt.grid()
    name = "out/" + savename
    plt.savefig(name+".pdf")

def Plot_preprocessing(df_raw, Feature_Selector, Scaler, Model,X):
    mpl.style.use('seaborn')
    titel = Feature_Selector + " " + Model
    plt.figure()
    for scl in Scaler:
        df = df_raw[df_raw['Feature_Selector'] == Feature_Selector]
        df = df[df['Scaler'] == scl]
        df = df[df["Model"] == Model]

        plt.grid
        label = scl + " Scaler"
        
        plt.plot(df.k,df[X],label = label)
        plt.xlabel("different k")
        plt.ylabel(X)
        plt.title(titel)
        plt.legend()
        save = "out/kBest|PCA/" + Model + "_with_different_k_sclaer_"+ X + Feature_Selector
        plt.savefig(save+".pdf")

def FindBestK_Scaler(df, model):
    df_best = df[ df['Model'] == model]
    df_best = df_best[ df_best["Score"] == max(df_best["Score"])]
    k = df_best["k"]
    scl = df_best["Scaler"]

    return k, scl

def FindBest_Params(df):
    return df[df.score == np.max(df.score)]
