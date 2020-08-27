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
from sklearn import preprocessing
from classifiers.bayesianKnn import BayesianKnnClassifier

classifier = BayesianKnnClassifier(k=5)

X = pd.read_csv("./data/mfeat-fac", sep=r'\s+')
y = pd.read_csv("./data/y_fac.csv", sep=";")

y = y.values

X_scaled = preprocessing.scale(X)
y_pred = classifier.classify(X, y)