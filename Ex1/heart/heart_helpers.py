import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.impute import KNNImputer

HEART_SAMPLES, HEART_NUM_COLS = 303, 14

HEART_FEATS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
HEART_TARGET = "target"


class heart_columns:
    explanations = {
        "age": "age",
        "sex": "sex",
        "cp": "chest pain type, Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic", 
        "trestbps": "resting blood pressure",
        "chol": "serum cholestoral in mg/dl",
        "fbs": "fasting blood sugar > 120 mg/dl",
        "restecg": "resting electrocardiographic results (values 0,1,2)",
        "thalach": "maximum heart rate achieved",
        "exang": "exercise induced angina",
        "oldpeak": "oldpeak = ST depression induced by exercise relative to rest",
        "slope": "the slope of the peak exercise ST segment",
        "ca": "number of major vessels (0-3) colored by flourosopy",
        "thal": "thalium Stress Test Result: 3 = normal; 6 = fixed defect; 7 = reversable defect",
        "target": "presence of heart disease valued 0, 1, 2, 3, 4"
        }
    order = ["age",
        "sex",
        "cp", 
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target"]
    
    def __repr__(self):
        return str(self.explanations)

    def __getitem__(self, col):
        if isinstance(col, str): return self.explanations[col]
        if isinstance(col, int): return self.order[col]

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


def process_heart(df, impute_mode = 0):
    """
    arguments:
        impute_mode...  0 = drop
                        1 = knn
                        2 = 0->1->2->3
    """

    df_ret = df.replace(to_replace="?", value=np.nan)
    if impute_mode == 0:
        df_ret.replace(to_replace={"thal": {"3": 0, "7": 1, "6": 2}}, value=None,inplace=True)
        df_ret.dropna(inplace=True)
        df_ret["ca"] = df_ret["ca"].astype("int64")
        df_ret["thal"] = df_ret["thal"].astype("int64")
        return df_ret

    elif impute_mode == 1:
        df_ret.replace(to_replace={
            "thal": {"3": 0, "7": 1, "6": 2},
            "ca": {"0": 0, "1": 1, "2": 2, "3":3}}, value=None,inplace=True)
        imputer = KNNImputer(n_neighbors=2)
        after = pd.DataFrame(imputer.fit_transform(df_ret), columns=HEART_FEATS+["target"])
        attr = "ca"
        after[attr+"Match"] = np.where(df_ret[attr] == after[attr], True, False)
        attr = "thal"
        after[attr+"Match"] = np.where(df_ret[attr] == after[attr], True, False)
        return after

    elif impute_mode == 2:
        df_ret.replace(to_replace={
            "thal": {"3": 0, "7": 2, "6": 3, np.nan: 1},
            "ca": {"0": 0, "1": 1, "2": 2, "3":3, np.nan: 0}}, value=None,inplace=True)
        df_ret["ca"] = df_ret["ca"].astype("int64")
        df_ret["thal"] = df_ret["thal"].astype("int64")
        return df_ret

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