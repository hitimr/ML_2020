import pandas as pd


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
    differences = pd.DataFrame()
    differences["different"] = df_a[col_name].eq(df_b[col_name])
    return differences["different"].value_counts(True)


