import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_CV_with_Std(df, y = "R2_score", regressor = "sklearn", titel = "score vs stabdartdeviation", SaveName = False):
    kmax  = np.max(df["k"])
    list_k = np.linspace(1, kmax, num=kmax)
    std_list = np.zeros(kmax)
    mean_list = np.zeros(kmax)
    for k in list_k:
        df_tmp = df[df["k"] == k]
        std_list[int(k-1)] = np.std(df_tmp[y])
        mean_list[int(k-1)] = np.mean(df_tmp[y])
    plt.plot(list_k, mean_list, '-', label = regressor)
    plt.fill_between(list_k, mean_list - std_list, mean_list + std_list, alpha=0.2)
    plt.grid()
    plt.legend()
    plt.xlabel("k-splits")
    plt.title(titel)
    plt.ylabel(y)
    if SaveName:
        plt.savefig(SaveName)