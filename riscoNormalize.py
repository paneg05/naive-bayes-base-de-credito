import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB

baseRiscoCredito = pd.read_csv('./databases/risco_credito.csv')
##print(baseRiscoCredito)

xRiscoCredito = baseRiscoCredito.iloc[:,0:4].values
##print(xRiscoCredito)

yRiscoCredito = baseRiscoCredito.iloc[:, 4].values
##print(yRiscoCredito)


labelEncoderHistoria = LabelEncoder()
labelEncoderDivida = LabelEncoder()
labelEncoderGarantia = LabelEncoder()
labelEncoderRenda = LabelEncoder()

xRiscoCredito[:, 0] = labelEncoderHistoria.fit_transform(xRiscoCredito[:, 0])
xRiscoCredito[:, 1] = labelEncoderDivida.fit_transform(xRiscoCredito[:, 1])
xRiscoCredito[:, 2] = labelEncoderGarantia.fit_transform(xRiscoCredito[:, 2])
xRiscoCredito[:, 3] = labelEncoderRenda.fit_transform(xRiscoCredito[:, 3])

##print(xRiscoCredito)

with open('./databases/riscoDeCredito.pkl', mode='wb') as f:
    pickle.dump([xRiscoCredito, yRiscoCredito], f)




