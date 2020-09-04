# [josé marcus]
# 
# 1) criar função para avaliação dos classificadores, envolvendo as etapas:
# - validação cruzada estratificada 
# "30 times ten fold" para avaliar e comparar os classificadores combinados. 
# - Se necessário, retire do conjunto de aprendizagem, um conjunto de validação para
# fazer ajuste de parametros e depois treine o modelo novamente com os conjuntos aprendizagem + validação.
# - Obtenha uma estimativa pontual e um intervalo de confiança para a taxa de acerto de cada classificador;
# - Usar o Friedman test (teste não parametrico) para comparar os classificadores;

import pandas as pd
import numpy as np
from sklearn import preprocessing
from classifiers.bayesianKnn import BayesianKnnClassifier
from classifiers.GaussianBayes import GaussianBayes
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def get_best_model_by_cross_validation(X_train, y_train, model, **kwargs):
    
    cv = RepeatedKFold(n_splits=10, n_repeats=30, random_state=1)

    clf = GridSearchCV(estimator=model, param_grid=kwargs, n_jobs=-1, cv=cv, scoring='accuracy')
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    return best_model

def train_test_validation(X, y, model, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=1)
    
    X_train_scaled = preprocessing.scale(X_train)

    classifier = get_best_model_by_cross_validation(X_train_scaled, y_train, model, **kwargs)

    X_test_scaled = preprocessing.scale(X_test)

    y_pred = classifier.predict(X_test_scaled)

    # cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # print("model confusion matrix\n", cm)
    print("model accuracy: ", acc)

views = ['fou', 'kar', 'fac']

model_names = ['bayesian_knn', 'gaussian_bayes']
models = [BayesianKnnClassifier(), GaussianBayes()]
kwargs = [{ 'k' : range(1, 12, 2)}, {}]

for model, params, model_name in zip(models, kwargs, model_names):
    for view in views:
        print('model: ', model_name)
        print('view: ', view)

        X_view = pd.read_csv(f'./data/mfeat-{view}', sep=r'\s+',  header= None)
        y_view = pd.read_csv(f'./data/particao_crispy.csv', sep=";", header= None)

        y_view = y_view.values[:, 0]
        train_test_validation(X_view, y_view, model, **params)
