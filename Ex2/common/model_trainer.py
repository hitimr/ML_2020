import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from .misc import plot_confusion_matrix

# New version for regression

class ModelTrainerReg():
    params = {}
    sklearn_model = object
    best_result = pd.DataFrame

    def __init__(self, sklearn_model, params:dict, x_train, y_train, x_test, y_test, f_eval=accuracy_score, thread_cnt=8):
        """Initialize the trainer

        Args:
            sklearn_model (object): Model from sklearn. for example KNNei...
            params (dict): dictionary of keywords. each entry must contain a list of parameters
            x_train (array-like): input data for training
            y_train (array-like): target data for training
            x_test (array-like): input data for testing
            y_test (array-like): target for testing
            f_eval (function): function for evaluating a given model that returns a value indicating the score of a model. the model with the highest score is picked
            thread_cnt (int. optional): number of threads for processing

        """        
        self.sklearn_model = sklearn_model
        self.params = params
        self.param_keys = list(params.keys())
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train).ravel() # not necessary but otherwise a warning might pop up
        self.y_test = np.array(y_test).ravel()
        self.f_eval = f_eval
        self.thread_cnt = thread_cnt

        self._eval_setup = {}


    def train(self):
        """Strart the training with multiple threads (thread_cnt)

        Returns:
            dict, float: dictionary containing best set of parameters, best score
        """        
        start_time = time()

        # reset previous results
        self.best_result = pd.DataFrame()
        self.result = pd.DataFrame()

        # Generate dictionaries of all posible parameter permutations
        keys, values = zip(*self.params.items())
        self.permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]    

        # Run through all models in parallel threads
        with Pool(self.thread_cnt) as p:
            result = p.map(self.analyze_model, self.permutations_dict)


        # wrap up results
        self.result = pd.DataFrame(result)
        self.best_result = self.result.iloc[self.result["score"].argmax()]  # store row with the best score
        end_time = time()
        print("Finished evaluation")
        print("Best parameteters found with:", self.best_parameter_set())
        print("score=", self.best_score())
        print("Total evaluation time = {:.2f}s".format(end_time-start_time))

        return self.best_parameter_set(), self.best_score()

    def analyze_model(self, parameter_set):
        model = self.sklearn_model(**parameter_set)
        start = time()
        model.fit(self.x_train, self.y_train)  # fit the model
        parameter_set["train_time"] = time() - start
        start = time()
        y_pred = model.predict(self.x_test)    # make prediction
        parameter_set["inference_time"] = time() - start

        parameter_set["score"] = self.f_eval(self.y_test, y_pred)   # add score to parameter set --> custom score given by f_eval
        # Add remaining scores
        for score, func in zip(["accuracy", "f1", "recall", "precision"], [accuracy_score, f1_score, recall_score, precision_score]):
            if score in self._eval_setup.keys():
                parameter_set[score] = func(self.y_test, y_pred, **self._eval_setup[score])
            else:
                if score == "accuracy":
                    parameter_set[score] = func(self.y_test, y_pred)
                else:
                    parameter_set[score] = func(self.y_test, y_pred, average='macro')

        return parameter_set

    @property
    def eval_setup(self, a_dict:dict):
        self._eval_setup = a_dict

    @eval_setup.setter
    def eval_setup(self, score:str, args):
        self._eval_setup[score] = args

    @eval_setup.getter
    def eval_setup(self):
        return self._eval_setup

    @eval_setup.getter
    def eval_setup(self, key:str):
        return self._eval_setup[key]

    def save_result(self, fileName):
        data = pd.DataFrame(self.result)
        data.to_csv(fileName, index=False)

    def best_parameter_set(self, dict_orient='dict'):
        #return self.best_result.drop("score").to_dict()
        par = self.result.iloc[self.result["score"].idxmax(),:]
        return par[self.param_keys].to_dict()

    def best_score(self, ret_index=False):
        #return self.best_result["score"]
        return self.result["score"].max(), self.result["score"].idxmax() if ret_index else self.result["score"].max()

    def worst_score(self, ret_index=False):
        return self.result["score"].min(), self.result["score"].idxmin() if ret_index else self.result["score"].min()

    def worst_parameter_set(self, dict_orient='dict'):
        par = self.result.iloc[self.result["score"].idxmin(),:]
        return par[self.param_keys].to_dict()


# Example for using the Model Trainer
if __name__ == "__main__":  
    # from sklearn.neighbors import KNeighborsClassifier
    # from sklearn.model_selection import train_test_split
    # from sklearn.metrics import accuracy_score
    # from sklearn.datasets import load_iris

    # params = {
    #     "n_neighbors" : list(range(1, 20)), 
    #     "weights" : ["uniform", "distance"],
    #     "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]
    #     }


    # data = load_iris()
    # x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

    # modeltrainer = ModelTrainer(KNeighborsClassifier, params, x_train, y_train, x_test, y_test, accuracy_score, thread_cnt=4)
    # modeltrainer.train()
    # modeltrainer.save_result("common/iris_result.csv")