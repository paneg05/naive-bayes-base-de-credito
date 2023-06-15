import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from yellowbrick.classifier import ConfusionMatrix

with open('./databases/credit.pkl','rb') as f:
    xCreditTreinamento= pickle.load(f)
    xCreditTeste = pickle.load(f)
    yCreditTreinamento = pickle.load(f)
    yCreditTeste  = pickle.load(f)


naiveCreditData = GaussianNB()
naiveCreditData.fit(xCreditTreinamento,yCreditTreinamento)

previsoes = naiveCreditData.predict(xCreditTeste)

cm= ConfusionMatrix(naiveCreditData)
cm.fit(xCreditTreinamento,yCreditTreinamento)
cm.score(xCreditTeste,yCreditTeste)

classificationReport = classification_report(yCreditTeste, previsoes)


