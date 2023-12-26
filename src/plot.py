import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def plot_tpr_fpr(dataset, subset, argv):
    tpr, fpr = [], []
    for arg in argv:
        df = pd.read_csv(f'../result/{arg}_{dataset}_{subset}.csv')
        tpr.append(df.loc[0, 'TPR'])
        fpr.append(df.loc[0, 'FPR'])
    plt.xticks = np.arange(1, len(argv) + 1)
    plt.plot(argv, tpr)
    plt.plot(argv, fpr)
    plt.legend()
    plt.savefig(f'tpr_fpr.png')


def plot_auc(dataset, subset, argv):
    pass


if __name__ == '__main__':
    dataset = sys.argv[1]
    subset = sys.argv[2]
    argv = sys.argv[3:]