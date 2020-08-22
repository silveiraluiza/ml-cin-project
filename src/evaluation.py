# [josé marcus]
# 
# 1) criar função para avaliação dos classificadores, envolvendo as etapas:
# - validação cruzada estratificada 
# "30 times ten fold" para avaliar e comparar os classificadores combinados. 
# - Se necessário, retire do conjunto de aprendizagem, um conjunto de validação para
# fazer ajuste de parametros e depois treine o modelo novamente com os conjuntos aprendizagem + validação.
# - Obtenha uma estimativa pontual e um intervalo de confiança para a taxa de acerto de cada classificador;
# - Usar o Friedman test (teste não parametrico) para comparar os classificadores;