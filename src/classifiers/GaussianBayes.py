class GaussianBayes:

  def __init__(self, X, y_train, K, C, d):
    self.X = X
    self.y_train = y_train
    self.K = K
    self.C = C
    self.d = d

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
  def estMaxVer(self, particao, n, C):
    Pwi = []
    for i in range(0, C):
      pwi = 0
      for exemplo_rotulado in range(0,len(particao[i])):
        pwi += 1
      pwi = pwi / n
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
  def matriz_covariancia(self, particao, variancia2, C, d):      # Função para criar a partir da variancia^2 a soma dos atributos por classe e dividir por n elementos na classe
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
  def prod_matriz(self, variancia, covariancia, C, K):
    classe = []
    for i in range(0, C):
      cl = []
      for k in range(0, K):
        resultado = (1 / ((2*math.pi**(self.d/2)) * (np.linalg.det(covariancia[i])**(1/2)))) * math.exp((-1/2) * variancia[i][k].T.dot(np.linalg.inv(covariancia[i]).dot(variancia[i][k])))
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
  def Predict(self, prob_normal_multivariada, P_wi, pr_clr_somatorio, C, K):
    lista_prob_post = []
    for i in range(0,C):
      lista_prob = []
      for k in range(0,K):
        lista_prob.append((prob_normal_multivariada[i][k] * P_wi[i]) / pr_clr_somatorio[k])
      lista_prob_post.append(lista_prob)
    lista_prob_post = np.array(lista_prob_post).T
    prediction = []
    for k in range(0,K):
      prediction.append(np.argmax(lista_prob_post[k]))
    # Lista com cada exemplo e sua respectiva classe a ser escolhida
    return prediction

  # TREINO
  def fit(self, X_train, y_train):
    # Encontrando a partição por exemplos dos grupos
    particao = self.crispy(y_train, self.C)
    
    # Calculando a matriz de maxima verossimilhança para o exemplo
    P_wi = self.estMaxVer(particao, len(X_train), self.C)
    
    # Calculando o vetor de médias para o exemplo
    media = self.medias(particao, self.C, self.d)
    
    # Sigma_ij para o calculo da probabilidade a posteriori. Usando as matrizes de média e a matriz de atributos X
    variancia1, variancia2 = self.variancia(X_train, media, self.C)

    covariancia = self.matriz_covariancia(particao, variancia2, self.C, self.d)
    
    # Definir as probabilidades utilizando uma Estimativa de Máxima Verossimilhança supondo uma Normal Multivariada
    prob_normal_multivariada = self.prod_matriz(variancia1, covariancia, self.C, len(X_train))
    
    pr_clr_somatorio = self.soma_termo_normalizacao(prob_normal_multivariada, P_wi, self.C, len(X_train))
    
    # Lista com cada exemplo e sua respectiva classe a ser escolhida
    y_pred = self.Predict(prob_normal_multivariada, P_wi, pr_clr_somatorio, self.C, len(X_train))
    return P_wi, media, covariancia, y_pred

  # Predição no TESTE
  def predict(self, X_test, P_wi, media, covariancia):
    # Sigma_ij para o calculo da probabilidade a posteriori. Usando as matrizes de média e a matriz de atributos X
    variancia1, variancia2 = self.variancia(X_test, media, self.C)
    
    # Definir as probabilidades utilizando uma Estimativa de Máxima Verossimilhança supondo uma Normal Multivariada
    prob_normal_multivariada = self.prod_matriz(variancia1, covariancia, self.C, len(X_test))
    
    pr_clr_somatorio = self.soma_termo_normalizacao(prob_normal_multivariada, P_wi, self.C, len(X_test))
    
    # Lista com cada exemplo e sua respectiva classe a ser escolhida
    y_pred = self.Predict(prob_normal_multivariada, P_wi, pr_clr_somatorio, self.C, len(X_test))
    return y_pred
