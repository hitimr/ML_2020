import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from time import time
from sklearn.metrics import confusion_matrix
from .misc import plot_confusion_matrix
from sklearn.metrics import f1_score as f1_score_eval

class ModelTrainer():
    params = {}
    sklearn_model = object
    best_result = pd.DataFrame

    def __init__(self, sklearn_model, params, x_train, y_train, x_test, y_test, f_eval, thread_cnt=8):
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
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train).ravel() # not necessary but otherwise a warning might pop up
        self.y_test = np.array(y_test).ravel()
        self.f_eval = f_eval
        self.sample_weights = None
        self.classes_names = None
        self.cms = None
        self.thread_cnt = thread_cnt


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
        if self.classes_names: # acts as trigger for computation of cms
            for i, dic in enumerate(result):
                dic["id"] = i
            self.cms = [(dic["id"] ,dic.pop("cm")) for dic in result]

        self.result = pd.DataFrame(result)
        self.best_result = self.result.iloc[self.result["score"].argmax()]  # store row with the best score
        self.best_result = self.result.iloc[self.result["f1_score"].argmax()]  # store row with the best score
        self.best_result = self.result.iloc[self.result["recall"].argmax()]  # store row with the best score
        self.best_result = self.result.iloc[self.result["precision"].argmax()]  # store row with the best score
        end_time = time()
        print("Finished evaluation")
        print("Best parameteters found with:", self.best_parameter_set())
        print("score=", self.best_score())
        #print("f1_score=", self.best_f1_score())
        #print("recall_score=", self.best_recall_score())
        #print("precision_score=", self.best_precision_score())
        print("Total evaluation time = {:.2f}s".format(end_time-start_time))

        return self.best_parameter_set(), self.best_score()

    def analyze_model(self, parameter_set):
        model = self.sklearn_model(**parameter_set)
        model.fit(self.x_train, self.y_train)  # fit the model
        y_pred = model.predict(self.x_test)    # make prediction
        score = self.f_eval(self.y_test, y_pred)        # used f_eval to evaluate score
        f1_score = f1_score_eval(self.y_test, y_pred, average='macro')
        cm = confusion_matrix(self.y_test, y_pred)
        recall = np.mean(np.diag(cm) / np.sum(cm, axis = 1))
        precision = np.mean(np.diag(cm) / np.sum(cm, axis = 0))

        parameter_set["score"] = score  # add score to parameter set
        parameter_set["f1_score"] = f1_score 
        parameter_set["recall"] = recall
        parameter_set["precision"] = precision

        if self.classes_names:
            cm = confusion_matrix(self.y_test, y_pred, labels=self.classes_names,sample_weight=self.sample_weights)
            parameter_set["cm"] = cm  # add score to parameter set

        return parameter_set

    def cm_setup(self, classes_names, sample_weights=None):
        self.classes_names = classes_names
        self.sample_weights = sample_weights

    def plot_confusion_matrix(self, id:int, title="Confusion matrix", cmap=plt.cm.Reds):
        """Plot a confusion matrix.

        Args:
            id...       id of cm to plot (can be selected from df)
            title...    title?
            cmap...     Colormap, should be one of matplotlibs, or the name of one

        Different colormaps can either be given directly, as above, or by name.


        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        return plot_confusion_matrix(self.cms[id][1], self.classes_names, normalize=True, title=title, cmap=cmap)


    def save_result(self, fileName):
        data = pd.DataFrame(self.result)
        data.to_csv(fileName, index=False)

    def best_parameter_set(self):
        return self.best_result.drop("score").to_dict()

    def best_score(self):
        return self.best_result["score"]
    
    def best_f1_score(self):
        return self.best_result["f1_score"]
    
    def best_recall_score(self):
        return self.best_result["recall"]

    def best_precision_score(self):
        return self.best_result["precision"]


# Example for using the Model Trainer
if __name__ == "__main__":  
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris
    from sklearn.metrics import f1_score

    params = {
        "n_neighbors" : list(range(1, 20)), 
        "weights" : ["uniform", "distance"],
        "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]
        }


    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

    modeltrainer = ModelTrainer(KNeighborsClassifier, params, x_train, y_train, x_test, y_test, accuracy_score, f1_score, thread_cnt=4)
    modeltrainer.train()
    modeltrainer.save_result("common/iris_result.csv")