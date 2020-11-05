import pandas as pd
import itertools


class ModelTrainer():
    params = {}
    sklearn_model = object
    best_score = 0
    best_parameter_set = {}

    def __init__(self, sklearn_model, params, x_train, y_train, x_test, y_test, f_eval):
        # add all parameters to self
        vars = locals()
        self.__dict__.update(vars)
        del self.__dict__["self"]
        self.sklearn_model = sklearn_model()

    def train(self):
        self.best_score = 0
        self.best_parameter_set = {}

        # Generate dictionaries of all posible parameter permutations
        keys, values = zip(*params.items())
        permutations_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Run through all permutations
        for parameter_set in permutations_dict:
            self.sklearn_model.__init__(**parameter_set)
            self.sklearn_model.fit(x_train, y_train)
            y_pred = self.sklearn_model.predict(x_test)
            score = self.f_eval(y_test, y_pred)
            
            if score > self.best_score:
                self.best_score = score
                self.best_parameter_set = parameter_set

        print("Finished evaluation")
        print("Best set of parameters is:", self.best_parameter_set)



# For development and testing...
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