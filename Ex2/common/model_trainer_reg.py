import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from time import time
from sklearn.metrics import accuracy_score ,mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

# New version for regression

class ModelTrainer():
    params = {}
    sklearn_model = object
    best_result = pd.DataFrame #tupel

    def __init__(self, sklearn_model, params:dict, row_data_data, row_data_target, Varerror = False, Error = False,learning_curve = learning_curve, f_eval=r2_score, rmse_eval = mean_squared_error,  CFeature = train_test_split, CV = KFold, LC = learning_curve,CV_shuffle=ShuffleSplit, thread_cnt=8):
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
        self.row_data_data = np.array(row_data_data)
        self.row_data_target = np.array(row_data_target)
        self.x_train = np.array(row_data_data)
        self.x_test = np.array(row_data_data)
        self.y_train = np.array(row_data_target).ravel() # not necessary but otherwise a warning might pop up
        self.y_test = np.array(row_data_target).ravel()
        self.CFeature = CFeature
        self.CV = CV
        self.LC = LC
        self.learning_curve = learning_curve
        self.Varerror = Varerror
        #self.y_row = np.array(y_test).ravel()
        self.f_eval = f_eval
        self.CV_shuffle = CV_shuffle
        self.rmse_eval = rmse_eval
        self.thread_cnt = thread_cnt
        self.Error = Error
        self.calc_cms = False
        self._cm_setup = {}
        self.cms = None

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
        print(keys, values)
        self.permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]    

        # Run through all models in parallel threads

        if(self.thread_cnt > 1):
            with Pool(self.thread_cnt) as p:
                result = p.map(self.analyze_model, self.permutations_dict)
        else:
            result = []
            for dik in self.permutations_dict:
                result.append(self.analyze_model(dik))
      


        # wrap up results
        self.result = pd.DataFrame(result)
        #self.best_result = self.result.iloc[self.result["R2_score"].argmax()]  # store row with the best score
        end_time = time()
        print("Finished evaluation")
        #print("Best parameteters found with:", self.best_parameter_set())
        print("R2_score=", self.best_score())
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

        parameter_set["R2_score"] = self.f_eval(self.y_test, y_pred)   # add score to parameter set --> custom score given by f_eval
        parameter_set["RMSE"] = self.rmse_eval(self.y_test, y_pred)
        if(self.Error):
            parameter_set[self.Error] = self.Varerror(self.y_test, y_pred)

        return parameter_set

    def CV_fold(self, k, test_size, random_state):
        kf = self.CV_shuffle(n_splits=k, test_size=test_size, random_state=random_state)
        kf.get_n_splits(self.row_data_data)

        for train_index, test_index in kf.split(self.row_data_data):
            self.x_train, self.x_test = self.row_data_data[train_index], self.row_data_data[test_index]
            self.y_train, self.y_test = self.row_data_target[train_index], self.row_data_target[test_index]
    

    def CV_shuffle_split(self, k, test_size, random_state):
        score_shuffle = []
        kf = self.CV_shuffle(n_splits=k, test_size=test_size, random_state=random_state)
        kf.get_n_splits(self.row_data_data)

        for train_index, test_index in kf.split(self.row_data_data):
            self.x_train, self.x_test = self.row_data_data[train_index], self.row_data_data[test_index]
            self.y_train, self.y_test = self.row_data_target[train_index], self.row_data_target[test_index]
            Best_set, Best_score = self.train()
            score_shuffle.append(Best_score)
        return score_shuffle

    def TTSplit(self, perc = 0.4, r_split = 42):
        self.x_train, self.x_test, self.y_train, self.y_test = self.CFeature(self.row_data_data, self.row_data_target, test_size=perc, random_state=r_split)

    def plot_learning_curve(self, model, title, axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        #model = self.sklearn_model(**parameter_set)
        train_sizes, train_scores, test_scores, fit_times = self.learning_curve(model(), self.row_data_data, self.row_data_target, cv=cv, n_jobs=self.thread_cnt, train_sizes=train_sizes, return_times=True)

        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1,
                                color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1,
                                color="g")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                        label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                        label="Cross-validation score")
        plt.legend(loc="best")
        plt.grid()
        return plt





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
        par = self.result.iloc[self.result["R2_score"].idxmax(),:]
        return par[self.param_keys].to_dict()

    def best_score(self, ret_index=False):
        #return self.best_result["score"]
        return self.result["R2_score"].max()#, self.result["score"].idxmax() if ret_index else self.result["score"].max()

    def worst_score(self, ret_index=False):
        return self.result["R2_score"].min(), self.result["R2_score"].idxmin() if ret_index else self.result["R2_score"].min()

    def worst_parameter_set(self, dict_orient='dict'):
        par = self.result.iloc[self.result["R2_score"].idxmin(),:]
        return par[self.param_keys].to_dict()


# Example for using the Model Trainer
if __name__ == "__main__":  
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split, ShuffleSplit
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score, mean_squared_log_error
    from sklearn.datasets import load_iris
    from sklearn.model_selection import ShuffleSplit
        

    params = {"alpha" : [0.0001]}
    params = {"weights" : ["uniform"]}
    params = {"n_estimators" : [100]}
    params = {"criterion": ["mse"]}
    
    data = load_iris()

    iris_data = data.data
    iris_target = data.target
    #x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    scaler.fit(iris_data)
    iris_data = scaler.transform(iris_data)

    title = "Learning_Curves_(SGD_Sklearn)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


    #modeltrainer = ModelTrainer(DecisionTreeRegressor, params, iris_data, iris_target,mean_squared_log_error, "RMLE",  r2_score)
    modeltrainer = ModelTrainer(DecisionTreeRegressor, params, iris_data, iris_target)#, r2_score)
    modeltrainer.TTSplit(perc = 0.4)
    modeltrainer.train()
    res = modeltrainer.result
    print(res)
    modeltrainer.CV_fold(k = 4)
    modeltrainer.train()
    res = modeltrainer.result
    print(res)
    socore_list = modeltrainer.CV_shuffle_split(k = 4, test_size = 0.4, random_state = 42)
    modeltrainer.train()
    res = modeltrainer.result
    print(res)
    #modeltrainer.plot_learning_curve(title = title, cv = cv)
    #modeltrainer.LC_plot()
    #modeltrainer.save_result("common/iris_result.csv")
    print("score_list", socore_list)