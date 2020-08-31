# @autor marcos de souza
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def normalize_and_split_data(X, y):
    X_scaled = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test




