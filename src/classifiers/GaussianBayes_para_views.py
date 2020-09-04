# imports
import numpy as np
import pandas as pd
import random
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import drive
import sys,os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importando os Views
dir_remoto = "/content/drive/My Drive/"
dir_local = os.getcwd() # path para rodar em máquina local ao invés do colab

fac = pd.read_csv(os.path.join(dir_remoto,'data/mfeat-fac'),  delim_whitespace=True, header= None)
fou = pd.read_csv(os.path.join(dir_remoto,'data/mfeat-fou'),  delim_whitespace=True, header= None)
kar = pd.read_csv(os.path.join(dir_remoto,'data/mfeat-kar'),  delim_whitespace=True, header= None)


# Partição Crispy para os três views simultaneamente
y = pd.read_csv(os.path.join('/content/drive/My Drive/Projeto I - AM/particao_crispy.csv'))
y = np.array(y)
y = y.reshape(1, -1)
y = y[0]


def train_test_validation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) #random_state=1
    print("y_train:",y_train)
    print("y_test:", y_test)

    K, d = len(X_train), len(X_train[0])    # Número de Exemplos e de dimensões
    C = 10
    classifier = GaussianBayes(X_train, y_train, K, C, d)
    P_wi, media, covariancia, y_predT = classifier.fit(X_train, y_train)
    print("y_predT:", y_predT)
    print("model accuracy (Train): ", accuracy_score(y_train, y_predT))

    K = len(X_test)            # Número de Exemplos
    y_pred = classifier.predict(X_test, P_wi, media, covariancia)
    print("y_pred:", y_pred)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("model confusion matrix\n", cm)
    print("model accuracy (Test): ", acc)


print("VIEW1: FAC")
X_scaled = preprocessing.scale(fac)
train_test_validation(X_scaled, y)

print("VIEW2: FOU")
X_scaled = preprocessing.scale(fou)
train_test_validation(X_scaled, y)

print("VIEW3: KAR")
X_scaled = preprocessing.scale(fac)
train_test_validation(X_scaled, y)
