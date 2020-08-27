# @author: marcos de souza
from sklearn.metrics import pairwise_distances
import numpy as np
from helpers import preprocessing as preprocess

class BayesianKnnClassifier:
    idx_neighbor = None
    y_neighbor = None

    def __init__(self, k):
        self.k = k
    
    
    def density(self, label, y):
        y_arr = [y[i][0] for i in range(0, len(y))]
        density = len([l for l in y_arr if l == label]) / len(y_arr)

        return density

    
    def posteriori(self, xi, y):
        # P(wj/xi) = P(xi/wj)*P(wj)
        n_classes = np.unique(xi)
        estimation = 0
        label = None

        for c in n_classes:
            likelihood = xi.count(c) / len(xi)
            c_density = self.density(c, y)
            c_estimation = likelihood * c_density

            if c_estimation > estimation:
                estimation = c_estimation
                label = c
        
        return estimation, label


    def classify(self, X, y):
        self.idx_neighbor, self.y_neighbor = preprocess.construct_knn_neighborhood(X, y, k=self.k)
        
        labels = []
        for xi in self.y_neighbor:
            estimation, label = self.posteriori(xi, y)
            print('obj: ', xi, 'estimation:', estimation, 'label: ', label)
            labels.append(label)

        return labels
