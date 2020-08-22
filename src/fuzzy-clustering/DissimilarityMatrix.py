import math
from typing import List
from random import *

# Example Data
E = 20 # Quantidade de Objetos
Y = 10 # Quantidade de variaveis

data = [] # Matriz de dados
for i in range(0, E):
    y = []
    for j in range(0, Y):
        y.append(randint(-100,100))
    data.append(y)
print(data)

# Distância Euclidiana
def dist_euclidiana(m1, m2):
    dim, soma = len(m1), 0
    for i in range(dim):
        soma += math.pow(m1[i] - m2[i], 2)
    return math.sqrt(soma)

# Matriz Lista com todos os valores em uma única lista
data2 = []
for linha in data:
    for col in linha:
        data2.append(col)
print("Cópia dos dados: ",data2)
Min = min(data2) # Recebe o menor valor entre os dados
Max = max(data2) # Recebe o maior valor entre os dados
Dif = (Max-Min) # Recebe a diferença entre os dados
print(Dif)

# Normalized Data with MaxMin
normalized = [] # Matriz com dados normalizados
for i in range(0,len(data)):
    norm_y = []
    for j in range(0,len(data[0])):
        norm_y.append((data[i][j] - Min) / (Dif))
    normalized.append(norm_y)
print(normalized)

def matrizDissimilaridade(matriz):
    linhas = len(matriz)
    matrizResultado = criarMatrizZerada(linhas)
    i = 0
    while i < linhas:
        j = 0
        while j < linhas:
            matrizResultado[i][j] = dist_euclidiana(matriz[i], matriz[j])
            j = j+1
        i = i+1
    return matrizResultado

def criarMatrizZerada(linhas:int) -> List[List[float]]:
    l = 0
    matrizResultado = []
    while l < linhas:
        matrizResultado.append([])
        c = 0
        while c < linhas:
            matrizResultado[l].append(0)
            c = c+1
        l = l+1
    return matrizResultado

print(matrizDissimilaridade(normalized))
