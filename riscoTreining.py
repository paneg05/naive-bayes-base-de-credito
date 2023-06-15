from sklearn.naive_bayes import GaussianNB
import pandas as pd

riscoCredito = pd.read_pickle('./databases/riscoDeCredito.pkl')
yRiscoCredito = riscoCredito[1]
xRiscoCredito = riscoCredito[0]

naiveRiscoCredito = GaussianNB()
naiveRiscoCredito.fit(xRiscoCredito, yRiscoCredito)

previsao = naiveRiscoCredito.predict([[0,0,1,2],[2,0,0,0]])
print(previsao)





