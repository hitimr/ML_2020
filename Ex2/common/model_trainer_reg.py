import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from time import time
from sklearn.metrics import r2_score ,mean_squared_error
from sklearn.model_selection import train_test_split, ShuffleSplit


# New version for regression

class ModelTrainer():
    params = {}
    sklearn_model = object

    def __init__(self, sklearn_model, params:dict, row_data_data, row_data_target, Variationerror = False, NameOfError = False, f_eval=r2_score, rmse_eval = mean_squared_error, CFeature = train_test_split, CV_shuffle=ShuffleSplit, thread_cnt=8):
        """Initialize the trainer

        Args:
            sklearn_model (object): Model from sklearn. for example KNNei...
            params (dict): dictionary of keywords. each entry must contain a list of parameters
            row_data_data (DateFrame): Feature data
            row_data_target (DateFrame): Target data
            Variationerror (function): additional errorfunction that you can choose 
            NameOfError (string): name of the additional errorfunction 
            x_train (array-like): input data for training
            y_train (array-like): target data for training
            x_test (array-like): input data for testing
            y_test (array-like): target for testing
            rmse_eval (function): function to calculate the mean_squraed_error
            CFeature (function): train-test-split from Skleran to seperate the data in test and validation set
            CV_shuffle (function): shuffel crossvalidation from Sklearn
            f_eval (function): function for evaluating a given model that returns a value indicating the score of a model.
            thread_cnt (int. optional): number of threads for processing

        """        
        self.sklearn_model = sklearn_model
        self.params = params
        self.param_keys = list(params.keys())
        self.row_data_data = np.array(row_data_data)
        self.row_data_target = np.array(row_data_target)
        self.x_train = np.array(row_data_data)
        self.x_test = np.array(row_data_data)
        self.y_train = np.array(row_data_target).ravel() # not necessary but otherwise a warning might pop up
        self.y_test = np.array(row_data_target).ravel()

        self.CFeature = CFeature
        self.CV_shuffle = CV_shuffle
        self.f_eval = f_eval 
        self.rmse_eval = rmse_eval

        self.Variationerror = Variationerror
        self.NameOfError = NameOfError

        self.CV_shuffle = CV_shuffle
        self.thread_cnt = thread_cnt

        self.k = False
        self.D = 1
        self.n_test = 1
        self.n_test = 1
        self.GolbalParameterSet = []



        self._eval_setup = {}


    def train(self):
        """Strart the training with multiple threads (thread_cnt)

        Returns:
            dict, float: dictionary containing best set of parameters, best score
        """        
        start_time = time()

        # Generate dictionaries of all posible parameter permutations
        keys, values = zip(*self.params.items())
        self.permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)] 
   

        # Run through all models in parallel threads

        if(self.thread_cnt > 1):
            with Pool(self.thread_cnt) as p:
                p.map(self.analyze_model, self.permutations_dict)
        else:
            result = []
            for dik in self.permutations_dict:
                self.analyze_model(dik)
      
        end_time = time()


    def analyze_model(self, parameter_set):
        model = self.sklearn_model(**parameter_set)
        start = time()
        model.fit(self.x_train, self.y_train)  # fit the model
        parameter_set["train_time"] = time() - start
        start = time()
        y_pred = model.predict(self.x_test)    # make prediction
        parameter_set["inference_time"] = time() - start
        if self.k:
            parameter_set["k"] = self.k
        parameter_set["R2_score"] = self.f_eval(self.y_test, y_pred)   # add score to parameter set --> custom score given by f_eval
        parameter_set["RMSE"] = self.rmse_eval(self.y_test, y_pred)
        parameter_set["D"] = self.D
        parameter_set["N"] = self.n_test + self.n_train
        parameter_set["n_test"] = self.n_test
        parameter_set["n_train"] = self.n_train

        if(self.NameOfError):
            parameter_set[self.NameOfError] = self.Variationerror(self.y_test, y_pred)

        self.GolbalParameterSet.append(parameter_set)


    def CV_shuffle_split(self, k, test_size, random_state):
        score_shuffle = []
        kf = self.CV_shuffle(n_splits=k, test_size=test_size, random_state=random_state)
        kf.get_n_splits(self.row_data_data)
        self.D = int(test_size * self.row_data_data.shape[1])
        

        for train_index, test_index in kf.split(self.row_data_data):
            self.x_train, self.x_test = self.row_data_data[train_index], self.row_data_data[test_index]
            self.y_train, self.y_test = self.row_data_target[train_index], self.row_data_target[test_index]
            self.n_test = len(self.y_test)
            self.n_train = len(self.y_train)
            self.train()
            self.k = self.k + 1

    def TTSplit(self, test_size = 0.4, r_split = 42):
        self.D = int(test_size * self.row_data_data.shape[1])
        self.x_train, self.x_test, self.y_train, self.y_test = self.CFeature(self.row_data_data, self.row_data_target, test_size=test_size, random_state=r_split)
        self.n_test = len(self.y_test)
        self.n_train = len(self.y_train)
        

    def retResults(self, fileName = False):  #returns the result dataframe and stores it in the given path
        results = pd.DataFrame()
        results = results.append(self.GolbalParameterSet, ignore_index=True)
        if fileName:
            results.to_csv(fileName, index=False)
        self.GolbalParameterSet = []
        self.k = 1
        return results

    def resetParameter(self):        # resets the GlobalParameterset to prevent overlapping between other tests
        self.GolbalParameterSet = []
    

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


# Example for using the Model Trainer
if __name__ == "__main__":  
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    from sklearn.preprocessing import StandardScaler

    from sklearn.model_selection import train_test_split, ShuffleSplit
    from sklearn.metrics import r2_score, mean_squared_log_error
    from sklearn.datasets import load_iris
    from sklearn.model_selection import ShuffleSplit
        

    params = {"alpha" : [0.0001]}        #Params for SGD
    #params = {"weights" : ["uniform"]}  #Params for KNN
    #params = {"n_estimators" : [100]}   #Params for RF
    #params = {"criterion": ["mse"]}     #Params for DT
    
    data = load_iris()

    iris_data = data.data
    iris_target = data.target


    scaler = StandardScaler()
    scaler.fit(iris_data)
    iris_data = scaler.transform(iris_data)

    test_size = 0.3
    n_splits = 3



    modeltrainer = ModelTrainer(SGDRegressor, params, iris_data, iris_target, thread_cnt=1)
    ########### train with TrainTestSplit  ###################
    modeltrainer.TTSplit(test_size = test_size)
    modeltrainer.train()
    results = modeltrainer.retResults()
    print(results)
    ############ shuffle_Cross validation  ###################
    modeltrainer.CV_shuffle_split(k = n_splits, test_size = test_size, random_state = 42)
    results = modeltrainer.retResults()
    print(results)