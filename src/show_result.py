import sys
import json
from argparse import ArgumentParser

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pprint import pprint

def show_result(res_list):
    fig, ax = plt.subplots()
    for res in res_list:
        for key,val in res[1].items():
            ax.plot(res[0], val, label=res[2]+'_'+key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.legend()
    ax.grid()
    ax.set_title("ResNet comparision")
    plt.show()


def show_confusion_matrix(y, gt, cl):
    cm = confusion_matrix(gt, y)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=cl, yticklabels=cl,
           title='resnet18_pretrained',
           ylabel='True label',
           xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white")
    fig.tight_layout()
    plt.show()


def main(args):
    source = {}
    files = args.file_name.split(',')
    result_list = []
    for fi in files:
        with open(fi, 'r') as f:
            source = json.load(f)
            if args.matrix:
                show_confusion_matrix(source['pred_y'], source['gt'], source['class'])
                return
        print(source['title'])
        pprint(list(zip(source['y_dict'].keys(), [max(i) for i in source['y_dict'].values()])))
        result_list.append([source['x'], source['y_dict'], source['title']])
    show_result(result_list)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    parser.add_argument("-mat", "--matrix", help="show Confusion Matrix", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()