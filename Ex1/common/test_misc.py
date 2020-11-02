import pytest

from misc import *


def test_compare_df():
    df_a = pd.DataFrame(np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [7, 8, 9, 10]]),columns=['a', 'b', 'c', 'd'])
    df_b = pd.DataFrame(np.array([[1, 20, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [7, 8, 9, 10]]),columns=['a', 'b', 'c', 'd'])

    print(df_a)
    print(df_b)
    assert compare_df(df_a, df_b, "a") == (1.0, 0.0)
    assert compare_df(df_a, df_b, "b") == (0.75, 0.25)
    return


if __name__ == "__main__":    
    test_compare_df()