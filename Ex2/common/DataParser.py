# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import config as cfg

import pandas as pd
import numpy as npß


def parse_test_housePrices(splitData=False):
    """Parse the testing data set "test_housePrices". This data is a copy from
    the leture of 11.11. PredictingNumericValues p.3

    Args: splitData (bool, optional): If true the data is split into x and y. Otherwise the complete Dataframe will be returnes. Defaults to False.

    Returns: [DataFrame or DataFrame, DataFrame]: The complete DataFrame or x and y
    """    
    df = pd.read_csv(cfg.DATASET_TEST_HOUSEPRICES, sep=";")
    if splitData:
        x = df.iloc[:,0:-1]
        y = df.iloc[:,[-1]]
        return x, y   # return complete Data Frame

    # Split
    return df

def parse_moneyball(splitData=False):
    """Parse the testing data set "moneyball". 

    Args: splitData (bool, optional): If true the data is split into x and y. Otherwise the complete Dataframe will be returnes. Defaults to False.

    Returns: [DataFrame or DataFrame, DataFrame]: The complete DataFrame or x and y
    """    
    df = pd.read_csv(cfg.DATASET_MONEYBALL, sep=",")

    if splitData:
        y = df[cfg.MONEYBALL_TARGET]
        x = df[cfg.MONEYBALL_FEATURES]
        return x, y   # split
    # whole
    return df

def parse_metro(splitData=False):
    """Parse the testing data set "metro". 

    Args: splitData (bool, optional): If true the data is split into x and y. Otherwise the complete Dataframe will be returnes. Defaults to False.

    Returns: [DataFrame or DataFrame, DataFrame]: The complete DataFrame or x and y
    """    
    df = pd.read_csv(cfg.DATASET_METRO, sep=",")

    if splitData:
        y = df[cfg.METRO_TARGET]
        x = df[cfg.METRO_FEATURES]
        return x, y   # split
    # whole
    return df


def parse_superconductivity(splitData=False):
    """Parse the testing data set "superconductivity". 

    Args: splitData (bool, optional): If true the data is split into x and y. Otherwise the complete Dataframe will be returnes. Defaults to False.

    Returns: [DataFrame or DataFrame, DataFrame]: The complete DataFrame or x and y
    """    
    df = pd.read_csv(cfg.DATASET_SUPERCONDUCTIVITY, sep=",")

    if splitData:
        y = df[cfg.SUPERCONDUCTIVITY_TARGET]
        x = df[cfg.SUPERCONDUCTIVITY_FEATURES]
        return x, y   # split
    # whole
    return df
