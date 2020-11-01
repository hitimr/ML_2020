import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize
from sklearn.feature_selection import VarianceThreshold, SelectKBest


def MLP_Search(alphas,modes,solv, h,maxiter,X_train, X_valid, Y_train, Y_valid,scaler):
    erg = []
    for g in h:
        for i in alphas:
            for j in modes:
                for k in solv:
                    for l in scaler:
                        X_train, X_valid = DefinScaler(l, X_train, X_valid)
                        clf = MLPClassifier(hidden_layer_sizes=(g,g,g,g), max_iter=maxiter, alpha=i,solver=k,activation=j,tol = 1e-9)
                        clf.fit(X_train, Y_train)
                        Y_pred = clf.predict(X_valid)
                        print("Hidden layers: ",g,"|\talpha: ",i,"|\tmode: ",j,"|\tsolver: ",k,"|\tscore: ",accuracy_score(Y_valid, Y_pred),"|\tscaler: ",l)
                        erg.append({
                            "h": g,
                            "alpha": i,
                            "mode": j,
                            "solver": k,
                            "score": accuracy_score(Y_valid, Y_pred),
                            "scaler": l})
        print("")
    return erg

def DefinScaler(scaler, X_train, X_valid):
    if scaler == "standard":
        scaler = StandardScaler()  
        scaler.fit(X_train) 
        X_train_SC = scaler.transform(X_train) 
        X_valid_SC = scaler.transform(X_valid)
    if scaler == "norm":
        X_train_SC = normalize(X_train, norm='l2')
        X_valid_SC = normalize(X_valid, norm='l2')
    if scaler == "minmax":
        min_max_scaler = MinMaxScaler()
        X_train_SC = min_max_scaler.fit_transform(X_train)
        X_valid_SC = min_max_scaler.fit_transform(X_valid)
    if scaler == "quantile":
        quantile_transformer = QuantileTransformer(random_state=0)
        X_train_SC = quantile_transformer.fit_transform(X_train)
        X_valid_SC = quantile_transformer.transform(X_valid)
  
    return X_train_SC, X_valid_SC

def DefineScaler(X_train, scaler_type):
    if scaler_type == "standard":
        scaler = StandardScaler()  
        scaler.fit(X_train)
        return scaler
    if scaler_type == "norm":
        scaler = NormalizeScaler().fit(X_train, norm='l2')
        return scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaler.fit_transform(X_train)
        return scaler
    if scaler_type == "quantile":
        scaler = QuantileTransformer(random_state=0)
        scaler.fit(X_train)
        return scaler

def FindBestScore(results):
    best_score = 0
    best_index = 0
    for i, res in enumerate(results):
        if res["score"] > best_score:
            best_score = res["score"]
            best_index = i
    print("best_score:", best_score)
    print("best_params:", results[best_index])
    return results[best_index]

def preprocess_rf(df, scaler=None, scaler_type="standard"):
    ...
    if not scaler:
        scaler = DefineScaler(df, scaler_type)
    df = scaler.fit(df)
    ...


def KBest(features, target):
    #VT = VarianceThreshold(threshold=0.05)
    #features = df.loc[:,'V1':'V10000']
    #target = df.loc[:,'Class']
    num_features = features.shape[1]
    #print(features)

    # Create and fit selector
    k= int(sqrt(num_features)*1.7)
    print(k)
    selector = SelectKBest(k=k)
    # Get columns to keep and create new dataframe with those only
    cols = selector.fit(features.values, target.values).get_support(indices=True)
    selected_feats = features.iloc[:,cols]
    return selected_feats

def plot_corr_heatmap(df, fmt=".2f", feat_to_ret="Class", ticksfont=12,abs = True):
    df = df.replace("republican",100)
    df = df.replace("democrat",1)
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    # Compute correlations and save in matrix
    if abs:
        corr = np.abs(df.corr()) # We only used absolute values for visualization purposes! ..."hot-cold" view to just sort between 
    else:
        corr = df.corr()

    # Mask the repeated values --> here: upper triangle

    #print(corr)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True # mask upper triangle

    corr_to_feat = corr.loc[:,feat_to_ret]
    
    f, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(corr, annot=True, fmt=fmt , mask=mask, vmin=0, vmax=1, linewidths=.5,cmap="YlGnBu")
    plt.tick_params(labelsize=ticksfont)
    return corr_to_feat

def Red_corr_list(corr_to_class_stripped,k):
    corr_list = corr_to_class_stripped.index
    red_corr_list = []
    for i in corr_list:
        if corr_to_class_stripped[i] <= k:
            red_corr_list.append(i)
    return red_corr_list

def Statistic(Y_valid,Y_pred,name):
    print("Heat map: ")
    plt.figure()
    cm = confusion_matrix(Y_valid.Class, Y_pred)
    sns.heatmap(cm, center=True)
    plt.savefig("Heatmap {}".format(name))
    plt.figure()
    sns.distplot(Y_valid.Class)
    sns.distplot(Y_pred, color="red")
    plt.savefig("difference between prediction and validation {}".format(name))

    plt.figure()
    sns.distplot(Y_valid.Class-Y_pred)
    plt.savefig("total difference between prediction and validation {}".format(name))
    print(sqrt(mean_squared_error(Y_valid.Class, Y_pred)))

    Y_pred_Norm = Y_pred / np.linalg.norm(Y_pred)
    Y_valid_Norm = Y_valid / np.linalg.norm(Y_valid.Class)

    print(sqrt(mean_squared_error(Y_pred_Norm, Y_valid_Norm)))