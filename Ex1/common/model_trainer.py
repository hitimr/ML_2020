import pandas as pd
import numpy as np
import itertools


class ModelTrainer():
    params = {}
    sklearn_model = object
    best_score = 0
    best_parameter_set = {}

    def __init__(self, sklearn_model, params, x_train, y_train, x_test, y_test, f_eval):
        """Set up the trainer

        Args:
            sklearn_model (object): Model from sklearn. for example KNNei...
            params (dict): dictionary of keywords. each entry must contain a list of parameters
            x_train (array-like): input data for training
            y_train (array-like): target data for training
            x_test (array-like): input data for testing
            y_test (array-like): target for testing
            f_eval (function): function for evaluating a given model that returns a value indicating the score of a model. the model with the highest score is picked
        """        
        self.sklearn_model = sklearn_model()
        self.params = params
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train).ravel() # not necessary but otherwise a warning might pop up
        self.y_test = np.array(y_test).ravel()
        self.f_eval = f_eval
    

    def train(self):
        self.best_score = 0
        self.best_parameter_set = {}

        # Generate dictionaries of all posible parameter permutations
        keys, values = zip(*self.params.items())
        self.permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Run through all permutations
        for parameter_set in self.permutations_dict:
            self.sklearn_model.__init__(**parameter_set)
            self.sklearn_model.fit(self.x_train, self.y_train)
            y_pred = self.sklearn_model.predict(self.x_test)
            score = self.f_eval(self.y_test, y_pred)
            parameter_set["score"] = score
            
            # Check if this model is better
            if score > self.best_score:
                self.best_score = score
                self.best_parameter_set = parameter_set

        print("Finished evaluation")
        print("Best set of parameters is:", self.best_parameter_set)
        return self.best_parameter_set

    def save_result(self, fileName):
        data = pd.DataFrame(self.permutations_dict)
        data.to_csv(fileName, index=False)


# Example for using the Model Trainer
if __name__ == "__main__":  
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import load_iris

    params = {
        "n_neighbors" : list(range(1, 20)), 
        "weights" : ["uniform", "distance"],
        "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]}


    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=1)

    modeltrainer = ModelTrainer(KNeighborsClassifier, params, x_train, y_train, x_test, y_test, accuracy_score)
    modeltrainer.train()
    modeltrainer.save_result("iris_result.csv")