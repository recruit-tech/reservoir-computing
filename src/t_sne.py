import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#%matplotlib inline

from sklearn.datasets import load_digits
import argparse

import pandas as pd




def read_acc_and_params_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna(0)



    accs = np.array(df['acc'] * 100, dtype='int64')

    params = df[df.columns[df.columns != 'acc']]

    print(accs)
    print(params)
    return accs, params


def main(csv_file):
    y, X = read_acc_and_params_from_csv(csv_file)

    ## load data
    #mnist = load_digits()
    #X = mnist.data
    #y = mnist.target

    print('X',X)
    print('y',y)
    print('X.shape',X.shape)
    print('y.shape',y.shape)

    ## t-SNE
    tsne = TSNE(n_components=2, random_state=1)
    tsne_reduced = tsne.fit_transform(X)

    ## PCA
    pca = PCA(n_components=2, random_state=1)
    pca_reduced = pca.fit_transform(X)

    ## Visualization
    plt.figure(figsize = (30,12))
    plt.subplot(121)
    plt.scatter(pca_reduced[:,0],pca_reduced[:,1], c = y, 
                edgecolor = "None", alpha=0.5)
    plt.colorbar()
    plt.title('PCA Scatter Plot')

    plt.subplot(122)
    plt.scatter(tsne_reduced[:,0],tsne_reduced[:,1],  c = y, 
            cmap = "coolwarm", edgecolor = "None", alpha=0.35)
    plt.colorbar()
    plt.title('TSNE Scatter Plot')
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
