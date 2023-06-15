import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

baseCredit = pd.read_csv('./databases/credit_data.csv')


##baseCredit2 = baseCredit.drop('age', axis= 1)
##baseCredit3 = baseCredit.drop(baseCredit[baseCredit['age'] < 0].index, axis= 1)

## sanitização dos dados
baseCredit.loc[baseCredit['age']<0, 'age'] = baseCredit['age'][baseCredit['age']>0].mean()
baseCredit.loc[pd.isnull(baseCredit['age']),'age'] = baseCredit['age'].mean()

##divisão entre classe e previsores
xCredit=baseCredit.iloc[:,1:4].values
yCredit=baseCredit.iloc[:,4].values

##escalonamento de dados
scaleCredit = StandardScaler()
xCredit= scaleCredit.fit_transform(xCredit)

xCreditTreinamento, xCreditTeste, yCreditTreinamento,yCreditTeste = train_test_split(xCredit,yCredit, test_size=0.25, random_state=0)


with open('./databases/credit.pkl','wb') as f:
    pickle.dump(xCreditTreinamento,f)
    pickle.dump(xCreditTeste,f)
    pickle.dump(yCreditTreinamento,f)
    pickle.dump(yCreditTeste,f)










