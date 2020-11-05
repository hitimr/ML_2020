import pandas as pd
import numpy as np
from  sklearn import preprocessing
from sklearn.impute import SimpleImputer


def preprocess(df):   
    # Formatting 
    df = df.astype('float') 
    data = df.iloc[:,2:-1]
    labels = df.iloc[:,[-1]]

    # Imputation
    imp = SimpleImputer(missing_values=np.NaN, strategy="median") 
    imp.fit(data)
    data = imp.transform(data)


    # Scaling
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(data_scaled)

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
