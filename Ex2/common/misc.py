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
