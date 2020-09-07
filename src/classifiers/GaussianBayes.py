import numpy as np
import math 
from sklearn.base import BaseEstimator, ClassifierMixin

class GaussianBayes(BaseEstimator, ClassifierMixin):

  # Encontrado a partir da partição crispy, uma lista com os exemplos por grupo
  def crispy(self, y_train, C):
    partition = []
    for i in range(0, self.C):
      part = []
      for j in range(1, len(y_train)+1):
        if i == y_train[j-1]:
          part.append(j)
      partition.append(part)
    return partition

  # Estimativa de Máxima Verossimilhança P(W_i):
  def estMaxVer(self, particao, C):
    Pwi = []
    for i in range(0, C):
      pwi = 0
      for _ in range(0, len(particao[i])):
        pwi += 1
      pwi = pwi / self.K
      Pwi.append(pwi)
    return Pwi

  # Vetor de médias mi_i para cada classe i pelo conjunto de atributos d
  def medias(self, particao, C, d):
    Medias_list = []
    for i in range(0, C):
      med_list = []
      N = len(particao[i])
      for n in range(0, d):
        med = 0
        for k in particao[i]:
          med += self.X[k-1][n]
        med_list.append(med/N)
      Medias_list.append(med_list)
    return np.array(Medias_list)

  # Sigma_ij para o calculo da probabilidade a posteriori. Usando as matrizes de média e a matriz de atributos X
  # Sigma_ij² para a matriz diagonal da covariancia. Então vamos fazer todo i=j, então só uma componente por vez.
  def variancia(self, X, media, C):
    variancia = []
    for j in range(0, C):  # 2 é o número de classes
      v = []
      for i in self.X:
        v.append(np.subtract(i, media[j]))
      variancia.append(v)
    variancia = np.array(variancia)
    variancia2 = variancia**2
    return variancia, variancia2

  # Matriz de Variância-Covariância (dxd)
  def matriz_covariancia(self, variancia2, particao, C, d):      # Função para criar a partir da variancia^2 a soma dos atributos por classe e dividir por n elementos na classe
    cov = []
    for i in range(0, C):
      v = []
      for j in range(0, d):  # j está relacionado ao número de d atributos/componentes
        soma = 0
        for pos in particao[i]:  # Anexar apenas os elementos que pertencem a mesma classe
          soma += (variancia2[i][pos-1][j])
        v.append(soma/len(particao[i]))
      cov.append(v)

    covariancia = np.zeros(shape=(C, d, d))   # Matriz Covariancia tem C-dimensões com cada dimensão quadratica dxd

    for i in range(0, C):
        for j in range(0, d):  # j está relacionado ao número de d atributos/componentes
          covariancia[i][j][j] = cov[i][j]
    return covariancia

  # Definir a probabilidade pela normal multivariada. Iterar por cada k (exemplo)
  # Definir as probabilidades utilizando uma Estimativa de Máxima Verossimilhança supondo uma Normal Multivariada
  def prod_matriz(self, variancia, covariancia, C, K, d):
    classe = []
    for i in range(0, C):
      cl = []
      for k in range(0, K):
        resultado = (1 / ((2*math.pi**(d/2)) * (np.linalg.det(covariancia[i])**(1/2)))) * math.exp((-1/2) * variancia[i][k].T.dot(np.linalg.inv(covariancia[i]).dot(variancia[i][k])))
        cl.append(resultado)
      classe.append(cl)
    return np.array(classe)

  def soma_termo_normalizacao(self, prob_normal_multivariada, P_wi, C, K):
    pr_clr_somatorio = []
    for k in range(0, K):
      s = 0
      for r in range(0, C):
        s += prob_normal_multivariada[r][k] * P_wi[r]
      pr_clr_somatorio.append(s)
    return pr_clr_somatorio

  # Predição: Afetar o exemplo X_k à classe W_i, se:
  def posteriori(self, prob_normal_multivariada, P_wi, pr_clr_somatorio, C, K):
    lista_prob_post = []
    for i in range(0, C):
      lista_prob_post.append([])
      for k in range(0, K):
        prob = (prob_normal_multivariada[i][k] * P_wi[i]) / pr_clr_somatorio[k]
        lista_prob_post[i].append(prob)
    lista_prob_post = np.transpose(lista_prob_post)
    class_estimation = []
    for k in range(0, K):
      class_est = []
      for c in range(0, C):
        class_est.append([c, lista_prob_post[k][c]])
      class_estimation.append(class_est)
    prediction = [] # Lista com cada exemplo e sua respectiva classe a ser escolhida
    for k in range(0, K):
      prediction.append(np.argmax(lista_prob_post[k]))
    return class_estimation, prediction

  # TREINO
  def fit(self, X_train, y_train):

    self.K, self.d = len(X_train), len(X_train[0])   # Número de Exemplos e de dimensões
    self.C = 10

    # Encontrando a partição por exemplos dos grupos
    particao = self.crispy(y_train, self.C)
    
    # Calculando a matriz de maxima verossimilhança para o exemplo
    self.P_wi = self.estMaxVer(particao, self.C)
    
    self.X = X_train
    # Calculando o vetor de médias para o exemplo
    self.media = self.medias(particao, self.C, self.d)

    # Sigma_ij para o calculo da probabilidade a posteriori. Usando as matrizes de média e a matriz de atributos X
    _, variancia2 = self.variancia(X_train, self.media, self.C)
    
    self.covariancia = self.matriz_covariancia(variancia2, particao, self.C, self.d)

  # Predição
  def predict(self, X):
    self.K, self.d = len(X), len(X[0])    # Número de Exemplos e de dimensões
    self.C = 10
    self.X = X
    # Sigma_ij para o calculo da probabilidade a posteriori. Usando as matrizes de média e a matriz de atributos X
    variancia1, _ = self.variancia(X, self.media, self.C)

    # Definir as probabilidades utilizando uma Estimativa de Máxima Verossimilhança supondo uma Normal Multivariada
    prob_normal_multivariada = self.prod_matriz(variancia1, self.covariancia, self.C, len(X), len(X[0]))
    
    pr_clr_somatorio = self.soma_termo_normalizacao(prob_normal_multivariada, self.P_wi, self.C, len(X))
    
    # Lista com cada exemplo e sua respectiva classe a ser escolhida
    predicoes, labels = self.posteriori(prob_normal_multivariada, self.P_wi, pr_clr_somatorio, self.C, len(X))
    self.estimations = predicoes
    return labels
