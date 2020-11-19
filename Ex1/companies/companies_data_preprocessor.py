import pandas as pd
import numpy as np
from  sklearn import preprocessing
from sklearn.metrics import confusion_matrix 
from sklearn.impute import SimpleImputer


def preprocess(df, **args): 
    ### Formatting 
    df = df.astype('float') 
    data = df.iloc[:,0:-1]
    labels = df.iloc[:,[-1]]

    return data, labels


def drop_cols(df, cutoff):
    row_cnt = df.shape[0]
    dropped_cols = []

    # Iterate over columns and calculate relative amount of missing data
    for col in df:
        missing_cnt = df[col].isna().sum()

        # drop col if above cutoff
        if missing_cnt / row_cnt > cutoff:

            dropped_cols.append(col)

    df = df.drop(dropped_cols, axis=1)
    print("Dropped", dropped_cols)
    return df


def calculate_score(y_test, y_pred):


    """ Confucsion Matrix:
    
                    actual solv.   actual bankrupt
    pred. solvent                    
    pred. bancrup                     
    
    """

    TP = - 1    # correct prediction of solvend company - we make a bit of money (negative cost)
    FP = 100    # predicted solvent but company will go bancrupt - we lose a lot of money
    FN = 1      # predicted bancrupt but company will stay solvent - we could have made a bit of money
    TN = 0      # correct prediction of company will go bancrupt - we neither make or lose money

    cost_mat = [
        [TP, FP],
        [FN, TN]
    ]

    conf_mat = confusion_matrix(y_test, y_pred)
    ideal_case =  (conf_mat[0][0] + conf_mat[0][1])*TP
    cost = conf_mat[0][0]*cost_mat[0][0] + conf_mat[1][1]*cost_mat[1][1] +\
        conf_mat[0][1]*cost_mat[0][1] + conf_mat[1][0]*cost_mat[1][0]

    score = cost/ideal_case
    return score
