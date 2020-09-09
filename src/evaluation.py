# função para avaliação dos classificadores, envolvendo as etapas:
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
from classifiers.GaussianKde import KDEClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def get_best_model_by_cross_validation(X_train, y_train, model, **kwargs):
    
    cv = RepeatedKFold(n_splits=10, n_repeats=30, random_state=1)

    clf = GridSearchCV(estimator=model, param_grid=kwargs, n_jobs=-1, cv=cv, scoring='balanced_accuracy')
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_

    # print('bandwidth: ', best_model.bandwidth)

    cv_results = [np.max(clf.cv_results_[f'split{i}_test_score']) for i in range(0, 300)]

    return best_model, cv_results

def train_test_validation(X_train, X_test, y_train, y_true, model, **kwargs):    
    X_train_scaled = preprocessing.scale(X_train)

    classifier, cv_results = get_best_model_by_cross_validation(X_train_scaled, y_train, model, **kwargs)

    X_test_scaled = preprocessing.scale(X_test)

    y_pred = classifier.predict(X_test_scaled)

    probs = classifier.estimations

    # cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # print("model confusion matrix\n", cm)
    return probs, acc, cv_results

def priori(y_train):   # Usado para o classificador Bayesiano Gaussiano
    p_wi = []
    n_classes = np.unique(y_train)

    for c in n_classes:
        p = len([i for i in y_train if i == c]) / len(y_train)
        p_wi.append(p)
        
    return p_wi

views = ['fou', 'kar', 'fac']

model_names = [
    'gaussian_kde',
    'bayesian_knn',
    'gaussian_bayes'
]
models = [
    KDEClassifier(),
    BayesianKnnClassifier(),
    GaussianBayes()
]
kwargs = [
    {"bandwidth": 10 ** np.linspace(0, 2, 100)}, 
    { 'k' : range(3, 9, 2)},
    {}
]

probs = []
y_crispy = pd.read_csv(f'./data/particao_crispy.csv', sep=";", header= None)
y_crispy = y_crispy.values[:, 0]

cv_views = []

for model, params, model_name in zip(models, kwargs, model_names):
    for view in views:
        X_view = pd.read_csv(f'./data/mfeat-{view}', sep=r'\s+',  header= None)

        X_train, X_test, y_train, y_true = train_test_split(X_view, y_crispy, test_size = 0.10, random_state=1)
        
        probs_view, acc, cv_results_view = train_test_validation(X_train, X_test, y_train, y_true, model, **params)

        cv_views.append(cv_results_view)

        probs.append(probs_view)

        print('model: ', model_name)
        print('view: ', view)
        print('view acc: ', acc)

    # sum rule
    new_y_pred = []
    for pview1, pview2, pview3 in zip(probs[0], probs[1], probs[2]):
        best_sum = -9999999
        label = None
        obj_probs = np.concatenate([pview1, pview2, pview3])

        n_classes = np.unique([int(k[0]) for k in obj_probs])
        
        p_wi = priori(y_train)
        for c in n_classes:
            c_sum = (-2 * p_wi[c]) + (np.sum([k[1] for k in obj_probs if k[0] == c]))
            
            if c_sum > best_sum:
                best_sum = c_sum
                label = c

        new_y_pred.append(label)
    
    new_acc = accuracy_score(y_true, new_y_pred)
    print('accuracy by sum rule: ', new_acc)

cv_results_df = pd.DataFrame(cv_views)

cv_results_df.T.to_csv('cv_results.csv', sep=';', index=False)

acuracias = {'bayesian_knn': [0.79, 0.63, 0.775, 0.855], 'gaussian_bayes':[0.805, 0.58, 0.735, 0.800], 'gaussian_kde':[0.835, 0.615, 0.82, 0.835]}
for model_name in model_names:
    print()
    print("Model:", model_name)
    for i in acuracias.items():
        if model_name == i[0]:    
          for acc in i[1]:
            print("Classificador:", views[i[1].index(acc)])
            proporcao = acc
            print("Estimativa Pontual: ", proporcao)
            print("Intervalo de Confiança: ", calc_intervalo_confiança(proporcao))

def calc_intervalo_confiança(proporcao):
    n = 2000    # Número de exemplos
    z = 1.96    # Para uma Confiança de 95%, a Tabela Z apresenta valor crítico igual a 1.96
    diff = z*np.sqrt(proporcao*(1-proporcao)/n)
    upper = round(proporcao - diff, 5)
    lower = round(proporcao + diff, 5)
    return (upper, lower)
