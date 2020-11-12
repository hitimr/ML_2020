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


def process_heart(df, impute_mode = 0, scaler=None):
    """
    arguments:
        impute_mode...  0 = drop
                        1 = knn
                        2 = 0->1->2->3
    """

    df_ret = df.replace(to_replace="?", value=np.nan)
    cols = df.columns
    feats = [c for c in cols if c != "target"]
    target = "target"
    del cols
    if impute_mode == 0:
        try:
            df_ret.replace(to_replace={"thal": {"3": 0, "7": 1, "6": 2}}, value=None,inplace=True)
        except:
            df_ret.replace(to_replace={"thal": {3: 0, 7: 1, 6: 2}}, value=None,inplace=True)
        df_ret.dropna(inplace=True)
        df_ret["ca"] = df_ret["ca"].astype("int64")
        df_ret["thal"] = df_ret["thal"].astype("int64")
        if scaler:
            df_ret[feats] = scaler.fit_transform(X=df_ret[feats])
        return df_ret

    elif impute_mode == 1:
        try:
            df_ret.replace(to_replace={
                "thal": {"3": 0, "7": 1, "6": 2},
                "ca": {"0": 0, "1": 1, "2": 2, "3":3}}, value=None,inplace=True)
        except:
            df_ret.replace(to_replace={
                "thal": {3: 0, 7: 1, 6: 2}}, value=None,inplace=True)
        imputer = KNNImputer(n_neighbors=2)
        after = pd.DataFrame(imputer.fit_transform(df_ret), columns=HEART_FEATS+["target"])
        if scaler:
            after[feats] = scaler.fit_transform(after[feats])
        attr = "ca"
        after[attr+"Match"] = np.where(df_ret[attr] == after[attr], True, False)
        attr = "thal"
        after[attr+"Match"] = np.where(df_ret[attr] == after[attr], True, False)
        return after

    elif impute_mode == 2:
        try:
            df_ret.replace(to_replace={
            "thal": {"3": 0, "7": 2, "6": 3, np.nan: 1},
            "ca": {"0": 0, "1": 1, "2": 2, "3":3, np.nan: 0}}, value=None,inplace=True)
        except:
            df_ret.replace(to_replace={
                "thal": {3: 0, 7: 2, 6: 3, np.nan: 1},
                "ca": {0: 0, 1: 1, 2: 2, 3:3, np.nan: 0}}, value=None,inplace=True)
        df_ret["ca"] = df_ret["ca"].astype("int64")
        df_ret["thal"] = df_ret["thal"].astype("int64")
        if scaler:
            df_ret[feats] = scaler.fit_transform(df_ret[feats])
        return df_ret
