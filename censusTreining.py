import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

with open('./databases/census.pkl','rb') as f:
    xCensusTreinamento, xCensusTeste, yCensustreinamento, yCensusTeste= pickle.load(f)


naiveCensus = GaussianNB()
naiveCensus.fit(xCensusTreinamento,yCensustreinamento)
previsoes = naiveCensus.predict(xCensusTeste)

## se não executar o escalonamento pode-se alcançar 70% de precisão
report2= classification_report(yCensusTeste,previsoes)

print(report2)
