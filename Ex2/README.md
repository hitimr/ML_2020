# Exercise 2: Regression

Group 8

Members:

> - Alexander LEITNER
> - Mario HITI
> - Peter HOLZNER

Language: Python (>3.7)

## Packages

We're also providing a pipfile (for pipenv), that tracked all of the packages we installed/used for this project.

Here is a list of all relevant packages:

- numpy = 1.19.4
- pandas = 1.1.4
- matplotlib = 3.3.3
- seaborn = 0.11.0 (Plotting)
- jupyterlab = 2.2.9
- scikit-learn = 0.23.2 (Machine Learning)

## Implementation of algorithms

Project structure

- common: contains various functions and classes used throughout all of notebooks and scripts.
- Datasets: contains the actual data of each dataset.
- KNN: contains the KNNRegressor (.py) and a notebook used for the runtime analysis of the KNN
- GD: contains the GradientDescent algorithm and our implementation of the LinearRegressor (sklearn equivalent SGD). Also contais notebooks showing a usage example and benchmarks.
- [Metro, Moneyball, Superconductivity]: our notebooks for data analysis, model training and evaluation. Also all output (plots, training results, ...) is saved in the folder out.
- test: unit tests for some of our classes and functions.
- config.py: contains various configuration parameters (e.g. paths to the datasets) for our project
- Pipfile: contains info about the necessary packages. Can be used with pipenv (if you use it) instead of a requirements-file.
