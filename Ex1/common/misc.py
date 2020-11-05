import pandas as pd
import numpy as np


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
