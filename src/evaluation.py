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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def cross_validation(X, y):
    model = BayesianKnnClassifier(k=5)
    cv = RepeatedKFold(n_splits=10, n_repeats=30, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1, error_score='raise')

    print('f-measure (avg):', np.mean(n_scores))

def train_test_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)
    
    classifier = BayesianKnnClassifier(k=5)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("model confusion matrix\n", cm)
    print("model accuracy: ", acc)

X = pd.read_csv("./data/mfeat-fac", sep=r'\s+',  header= None)
y = pd.read_csv("./data/y_fac.csv", sep=";")

y = y.values[:, 0]

X_scaled = preprocessing.scale(X)
train_test_validation(X_scaled, y)

cross_validation(X_scaled, y)