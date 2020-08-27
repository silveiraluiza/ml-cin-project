# @autor marcos de souza
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

def normalize_and_split_data(X, y):
    X_scaled = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


def construct_knn_neighborhood (X, y, k):
    # compute euclidean distances
    D = pairwise_distances(X)
    D **= 2
    # sort the distance matrix D in ascending order
    idx = np.argsort(D, axis=1)
    idx_nn = idx[:, 1:k+1]

    [idx_nn[0][i] for i in range(0, k)]

    y_nn = []
    for obj in range(0, len(idx_nn)):
        y_nn.append([])
        for nn in idx_nn[obj]:
            label = [y[i][0] for i in range(0, len(y)) if y[i][1] == nn+1][0]
            y_nn[obj].append(label)
    return idx_nn, y_nn



