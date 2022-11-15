import sklearn.datasets
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize


def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


def read_acc_and_params_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna(0)

    accs = np.array(df['acc'] * 100, dtype='int64')

    df = df[df.columns[df.columns != 'acc']]
    df = df[df.columns[df.columns != 'fb_scale']]
    params = (df - df.min()) / (df.max() - df.min())

    print(accs)
    print(params)
    return accs, params.to_numpy()


def main(csv_file):

    accuracy, parameters = read_acc_and_params_from_csv(csv_file)

    print(parameters)
    print(type(parameters))

    norm = Normalize(vmin=0.95*min(accuracy), vmax=1.05*max(accuracy))


    parameters3d = TSNE(n_components=3, random_state=1).fit_transform(parameters)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10)).gca(projection='3d')

    max_acc = np.max(accuracy)
    for i in np.unique(accuracy):
        target = parameters3d[accuracy == i]
        if i != max_acc:
            fig.scatter(target[:, 0], target[:, 1], target[:, 2], c=accuracy[accuracy == i], norm=norm, cmap='jet', marker='.', alpha=0.5)
        else:
            fig.scatter(target[:, 0], target[:, 1], target[:, 2], s=100, c='black', marker='^')

    plt.show()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('-csv_file', dest='csv_file', type=str, help='result of grid search', required=True)

    params = parser.parse_args()
    csv_file = params.csv_file

    try:
        main(csv_file)
    except KeyboardInterrupt:
        print('closeing...')
    exit()
