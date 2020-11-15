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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer,normalize
from sklearn.feature_selection import VarianceThreshold, SelectKBest

from sklearn.metrics import f1_score
import time




def MLP_Search(alphas,modes,solv, h,maxiter,X_train, X_valid, Y_train, Y_valid, level):
    erg = []
    if level == 1:
        g = (h)
    if level == 2:
        g = (h,h)
    if level == 3:
        g = (h,h,h)
    if level == 4:
        g = (h,h,h,h)
    start = time.time()
    clf = MLPClassifier(hidden_layer_sizes=g, max_iter=maxiter, alpha=alphas,solver=solv,activation=modes,tol = 1e-9)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_valid)
    end = time.time()
    train_time = end - start
    f1, recall, precision = Scores(Y_valid, Y_pred)
    print("Hidden layers: ",h,"|\talpha: ",alphas,"|\tmode: ",modes,"|\tsolver: ",solv,"|\tscore: ",accuracy_score(Y_valid, Y_pred))
    erg = {
        "h": h,
        "alpha": alphas,
        "mode": modes,
        "solver": solv,
        "score": accuracy_score(Y_valid, Y_pred),
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "time": train_time}
    return erg

def Knn_Search(max_k,X_train, X_valid, Y_train, Y_valid):
    score_list = []
    erg = []
    start = time.time()
    for knn in range(1, max_k):
        model = KNeighborsClassifier(n_neighbors=knn).fit(X_train, Y_train)
        Y_pred = model.predict(X_valid)
        score = accuracy_score(Y_valid, Y_pred)
        score_list.append(score)
    end = time.time()
    train_time = (end - start)/max_k
    knn = score_list.index(max(score_list))+1
    score = max(score_list)
    f1, recall, precision = Scores(Y_valid, Y_pred)
    erg = {
        "k_max": knn,
        "score": score,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "time": train_time}
    return erg, score_list

    


def RM_Search(n_estimators, criterion, max_features, X_train, X_valid, Y_train, Y_valid):
    erg = []
    start = time.time()
    clf = RFC(n_estimators=n_estimators,criterion = criterion, max_features = max_features)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_valid)
    end = time.time()
    train_time = (end - start)
    score = accuracy_score(Y_valid, Y_pred)
    f1, recall, precision = Scores(Y_valid, Y_pred)
    erg = {
        "score": score,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "time": train_time}
    return erg


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


def Scores(Y_valid, Y_pred):
    f1 = f1_score(Y_valid, Y_pred, average='macro')
    cm = confusion_matrix(Y_valid, Y_pred)
    recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
    precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))

    return f1, np.mean(recall), np.mean(precision)