import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import itertools

# DEFAULT PARAMS FOR plot_params function
LOGX = True
YLIMS = (0.0, 1)
# Example params for RFC, can be overwritten
params = {
    "n_estimators": [1, 8, 10, 12, 15, 20, 50, 100,  1000],
    "max_features": ["sqrt", "log2"],
    "criterion": ["gini", "entropy"]
}

def plot_params(results, scores="score", fileName=None, params=params, ylims=YLIMS, logx=LOGX):
    """
    Plots results for different params. Works well in conjunction with the modeltrainer...will probably be included in the future.

    Formatting and order of params_dicts is import, see examples below:

    params_rf = {
        "n_estimators": [1, 8, 10, 12, 15, 20, 50, 100,  1000],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"]
    }

    params_knn = {
    "n_neighbors" : list(range(3,50)), 
    "weights" : ["uniform", "distance"],
    "p" : [1,2]
    }

    Note that the first param/key in the dict supplies the x-axis for the plots.
    """
    param_keys = list(params)
    first_key = param_keys[0]
    rest = param_keys[1:]

    plt.style.use('seaborn')
    if isinstance(scores, str):
        fig, ax = plt.subplots(figsize=(8,6))
        for vals in tuple(itertools.product(*tuple(x for x in tuple(params.values())[1:]))):
            label = " / ".join([str(x) for x in vals])
            filters = " & ".join([str(x)+' == "'+str(v)+'"' for x, v in zip(rest, vals)])
            results.query(filters).plot(
                x=first_key, y=scores, label=label,
                ax=ax, marker="o", logx=logx)
        plt.legend()
        ax.set_title(scores, fontsize=18)

        plt.ylim(*ylims)
        if fileName:
            plt.savefig(fileName)
        plt.show()
        return plt.gcf()

params_mlp = {
    "alpha" : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
    "hidden_layer_sizes" : [(50,50), (5,5), (5,50), (50,5)],
    #"solver" : ["adam","lbfgs"],
    "activation" : ["tanh", "relu"]
    }

def plot_mlp(results, scores="score", fileName=None, params=params_mlp, ylims=YLIMS):
    param_keys = list(params)
    first_key = param_keys[0]
    hls = param_keys[1]
    rest = param_keys[2:]
    res_by_hls = [ (results[results[hls]==hls_val], hls_val) for hls_val in params[hls]]
    #display(res_by_hls)

    plt.style.use('seaborn')
    if isinstance(scores, str):
        fig, ax = plt.subplots(figsize=(8,6))
        for vals in tuple(itertools.product(*tuple(x for x in tuple(params.values())[2:]))):
            filters = " & ".join([str(x)+' == "'+str(v)+'"' for x, v in zip(rest, vals)])
            for line, hls_val in res_by_hls:
                label = " / ".join([str(x) for x in vals]) + f" / {hls_val}"
                ls = "-" if "tanh" in vals else "--"
                line.query(filters).plot(
                    x=first_key, y=scores, label=label,
                    ax=ax, marker="o", logx=LOGX, linestyle=ls)
        plt.legend()
        ax.set_title(scores, fontsize=18)

        plt.ylim(*ylims)
        if fileName:
            plt.savefig(fileName)
        plt.show()
        return plt.gcf()

# Confusion matrix 
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Reds) :
    """classes are the possible classes, so e.g ["B","M"], s.t. the ordering matches the encoding"""
    plt.rcParams.update({'font.size': 12})
    num_samples = 1
    if normalize:
        num_samples = np.sum(cm)
    #print("#",num_samples)
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    # itertools.product() gives all combinations of the iterables
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        string = cm[i, j]
        if normalize:
            string /= num_samples
            string = f"{string:.2f}"
        plt.text(j, i, string, horizontalalignment = "center", color="black", backgroundcolor="white")#= "white" if cm[i, j] > thresh else "black", )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return plt.gcf()

def plot_corr_heatmap(df, fmt=".2f", feat_to_ret="Class", ticksfont=12, abs = True):
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    # Compute correlations and save in matrix
    if abs:
        corr = np.abs(df.corr()) # We only used absolute values for visualization purposes! ..."hot-cold" view to just sort between 
    else:
        corr = df.corr()

    # Mask the repeated values --> here: upper triangle

    #print(corr)
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True # mask upper triangle

    corr_to_feat = corr.loc[:,feat_to_ret]
    
    f, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(corr, annot=True, fmt=fmt , mask=mask, vmin=0, vmax=1, linewidths=.5,cmap="YlGnBu")
    plt.tick_params(labelsize=ticksfont)
    return corr_to_feat
