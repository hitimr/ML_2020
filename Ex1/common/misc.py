import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools


def compare_df(df_a, df_b, col_name):
    """Compares a df_a[col_name] with df_b[col_name]

    Args:
        df_a (pd.DataFrame): first DataFrame
        df_b (pd.DataFrame): second DataFrame
        col_name (str): name of the column

    Returns:
        returns: A pd.DataFrame with the percentage of correct and incorrect values

            True     % of identical entries
            False    % of nonidentical entries
            Name: different, dtype: float64
    """
    assert df_a.shape[0] == df_b.shape[0] # row count must be equal

    size = df_a.shape[0]
    identical_cnt = np.sum(df_a[col_name].to_numpy() == df_b[col_name].to_numpy())
    return identical_cnt / float(size), (size - identical_cnt) / float(size)

# Confusion matrix 
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Reds) :
    """classes are the possible classes, so e.g ["B","M"], s.t. the ordering matches the encoding"""
    plt.rcParams.update({'font.size': 12})
    num_samples = 1
    if normalize:
        num_samples = np.sum(cm)
    #print("#",num_samples)
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    # itertools.product() gives all combinations of the iterables
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        string = cm[i, j]
        if normalize:
            string /= num_samples
            string = f"{string:.2f}"
        plt.text(j, i, string, horizontalalignment = "center", color="black", backgroundcolor="white")#= "white" if cm[i, j] > thresh else "black", )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return plt.gcf()


def count_missing_vals(df):   
    # Helopful ressource: https://towardsdatascience.com/how-to-check-for-missing-values-in-pandas-d2749e45a345
    return df.isna().sum().sum()


def create_out_dir():
    import os
    if not os.path.exists('out'):
        os.makedirs('out')