pip install scikit-posthocs
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Teste de Friedman
# Primeiro criamos um dataframe com as views 0: fac, 1: fou, 2: kar e a linha 3 com a regra da soma.
# As colunas são bg: Bayesiano Gaussiano, bkv: Bayesiano K-vizinhos e Par: Parzen
tbacc = pd.DataFrame.from_dict({'view': {0: 0, 1: 1, 2: 2, 3: 3}, 'bg': {0: 0.805, 1: 0.58, 2: 0.735, 3: 0.815},
                                'bkv': {0: 0.79, 1: 0.63, 2: 0.775, 3: 0.855}, 'par': {0: 0.835, 1: 0.615, 2: 0.82,
                                3: 0.835}})
print(tbacc)
# Aplicamos o teste de Friedman sobre as acurácias dos três classificadores
result = friedmanchisquare(tbacc["bg"], tbacc["bkv"], tbacc["par"])
print(result)

# Em seguida, em caso de rejeição da hipótese Nula, fazemos o teste posthoc nemenyi
dados = pd.DataFrame.from_dict({'blocks': {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6:
2, 7: 3, 8: 0, 9: 1, 10: 2, 11: 3}, 'groups': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1,
8: 2, 9: 2, 10: 2, 11: 2},'y': {0: 0.805, 1: 0.58, 2: 0.735, 3: 0.815,
4: 0.79, 5: 0.63, 6: 0.775, 7: 0.855, 8: 0.835, 9: 0.615, 10: 0.820, 11: 0.835}})

print(dados)

sp.posthoc_nemenyi_friedman(dados, y_col='y', block_col='blocks', group_col='groups', melted= True)
