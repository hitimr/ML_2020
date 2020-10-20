import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# required for importin modules from other directories
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import common
from common.data_parser import *


def generate_histogram(df):
    sns.histplot(df, x="class")
    plt.xticks(ticks=[0,1], labels=["not bancrupt", "bancrupt"])
    plt.title("")
    plt.show()




companies_df = parse_companies()
generate_histogram(companies_df)


