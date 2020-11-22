import pandas as pd
import numpy as np
import seaborn as sns
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

def count_missing_vals(df):   
    # Helopful ressource: https://towardsdatascience.com/how-to-check-for-missing-values-in-pandas-d2749e45a345
    return df.isna().sum().sum()

def count_values(df, cols_to_count=[], mode=False, norm=False, print_counts=True):
    """Count occurence of values columnwise.

    Arguments:
        cols_to_count... list of columns names to not count
        mode... boolean, whether or not to interpret cols_to_count argument as 
                columns to count (mode=True), or columns to not count (mode=False, def)
        norm... normalize counts within column -> probabilities
        print_counts... print results to console
    """ 
    column_names = df.columns
    return_dict = {}
    if mode:
        for col in cols_to_count:
            counts = df[col].value_counts(normalize=norm)
            if print_counts:
                print(f"--- {col} ---")
                display(counts)
                print("#"*40)
            return_dict[col] = counts
        return return_dict

    cols_to_count2 = [c for c in column_names if c not in cols_to_count]
    for col in cols_to_count2:
        counts = df[col].value_counts(normalize=norm)
        if print_counts:
            print(f"--- {col} ---")
            display(counts)
            print("#"*40)
        return_dict[col] = counts
    return return_dict


def create_out_dir():
    import os
    if not os.path.exists('out'):
        os.makedirs('out')



######## THE FOLLOWING IS TO BE MOVED TO plotting.py ##########

def plot_corr_heatmap(df, fmt=".2f", feat_to_ret="Class", ticksfont=12, abs = True):
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

######## THE PREVIOUS IS TO BE MOVED TO plotting.py ##########