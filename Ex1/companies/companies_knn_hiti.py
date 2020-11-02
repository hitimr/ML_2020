#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy, time, random, time
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.data_parser import *
from common.misc import *


# In[2]:


def preprocess(df):
    
    df = drop_cols(df, 0.05)

    df = df.replace(np.NaN, 0)
    df = df.replace("b'0'", 0)
    df = df.replace("b'1'", 1)
    print("Number of missing values:", count_missing_vals(df))
    return df


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


def replace_missing_vals(df):
    for col in df:
        for entry in df[col]:
            if entry != True:
                df[col][entry] = 0
    return df


# In[3]:


def count_missing_vals(df):   
    # Helopful ressource: https://towardsdatascience.com/how-to-check-for-missing-values-in-pandas-d2749e45a345
    #print(df.isna())
    return df.isna().sum().sum()


# In[4]:


df = parse_companies(5)


df = preprocess(df)


# In[5]:


#data = df.iloc[:,[2,3,4,5]]
data = df.iloc[:,0:-2]  # remove class col
labels = df.iloc[:,[-1]]    # only class col
x_train, x_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=1 )



# In[6]:


print(y_train)


# In[7]:


max_k = 2
for k in range(1, max_k):
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    result = pd.DataFrame()
    result["class"] = model.predict(x_test)
    print("k = %s Accuracy = %s" % (k, compare_df(result, y_test, "class")[0]))


# In[ ]:




