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


def update_annot(ind):
    cmap = plt.cm.jet

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    #print([org_parameters[n][:] for n in ind["ind"]])
    #print([org_accuracy[n] for n in ind["ind"]])
    text = ''
    for acc, params in [(org_accuracy[n], org_parameters[n][:]) for n in ind["ind"]]:
        print(acc, params)
        text += str(acc) + ': ' + ','.join([str(i) for i in params]) + '\n'
    print(text)
    annot.set_text(text)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def read_acc_and_params_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df = df.fillna(0)

    org_accs = df['acc']
    accs = np.array(df['acc'] * 100, dtype='int64')

    df = df[df.columns[df.columns != 'acc']]
    org_params = df[df.columns[df.columns != 'fb_scale']]
    params = (org_params - org_params.min()) / (org_params.max() - org_params.min())
    params = params.fillna(0)

    #print(org_accs)
    #print(org_params)
    #print(accs)
    #print(params)
    return accs, params.to_numpy(), org_accs, org_params.to_numpy()


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('-csv_file', dest='csv_file', type=str, help='result of grid search', required=True)

    params = parser.parse_args()
    csv_file = params.csv_file

    accuracy, parameters, org_accuracy, org_parameters = read_acc_and_params_from_csv(csv_file)

    #print(parameters)
    #print(type(parameters))

    norm = Normalize(vmin=0.95*min(accuracy), vmax=1.05*max(accuracy))


    parameters3d = TSNE(n_components=3, random_state=1).fit_transform(parameters)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    sc = ax.scatter(parameters3d[:, 0], parameters3d[:, 1], parameters3d[:, 2], c=accuracy, norm=norm, cmap='jet', marker='.', alpha=0.5)

    fig.colorbar(sc, ax=ax, location='left')

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
    
    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

