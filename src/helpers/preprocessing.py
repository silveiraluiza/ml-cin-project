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


def construct_knn_neighborhood (X):
    # compute euclidean distances
    dist = pairwise_distances(X)
    dist **= 2
    # sort the distance matrix D in ascending order
    dump = np.sort(dist, axis=1)

    return dump



