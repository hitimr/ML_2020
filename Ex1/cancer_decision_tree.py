from sklearn import tree

# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from config import *
from common.data_parser import *



df = parse_breastCancer("lrn")

X = df.drop(df.columns[[0, 1]], axis=1)
Y = df.drop(df.columns[[i for i in range(2, df.shape[1])]], axis=1)
Y = Y.drop("ID", axis=1)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
tree.plot_tree(clf) 