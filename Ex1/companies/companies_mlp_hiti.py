#!/usr/bin/env python
# coding: utf-8

# In[5]:


# General imports
import numpy as np

# Data Analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Custom Stuff
# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from common import data_parser, misc
from common.model_trainer import ModelTrainer
import companies_data_preprocessor


# # Data Preparation

# In[6]:


df = data_parser.parse_companies(5)
data, labels = companies_data_preprocessor.preprocess(df, MinMaxScaling=True, imputation=1)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=1 )

params = {
    "hidden_layer_sizes" : [(100,)], 
    "alpha" : [0.0001, 0.0002]}

modeltrainer = ModelTrainer(MLPClassifier, params, x_train, y_train, x_test, y_test, accuracy_score)
modeltrainer.train()

# # Model Training

# In[8]:


best_params_SC = {}
best_params_SC["h"] = (100, )
best_params_SC["alpha"] = 0.0001
best_params_SC["mode"] = "relu"
best_params_SC["solver"] = "adam"


clf = MLPClassifier(
    hidden_layer_sizes=(best_params_SC["h"]), 
    alpha=best_params_SC["alpha"], 
    activation=best_params_SC["mode"],
    solver=best_params_SC["solver"],
    max_iter=2000)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


# # Evaluation

# In[ ]:


confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)
companies_data_preprocessor.calculate_score(confusion_mat)

