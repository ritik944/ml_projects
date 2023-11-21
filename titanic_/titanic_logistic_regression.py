
import pandas as pd
titanic =pd.read_csv(r'C:\Users\ritik\OneDrive\Documents\GitHub\jetson_nano_projects\titanic_\train.csv')

print(titanic.head())

titanic.tail()

titanic.info()

titanic.describe()

#process data
#as pclass is 3 then means their are 3 ports (assumed)

ports=pd.get_dummies(titanic.Embarked,prefix='Embarked')
ports.head()

titanic = titanic.join(ports)
titanic.drop( ['Embarked'] ,axis=1, inplace=True)
titanic.head()

#transform gender
titanic.Sex=titanic.Sex.map({'male':0,'female':1})
titanic.head()

#extract the target variable
y=titanic.Survived.copy()
x=titanic.drop(['Survived'],axis=1)
x=x.drop(['Name'],axis=1)
x=x.drop(['Ticket'],axis=1)
x=x.drop(['Cabin'],axis=1)
x=x.drop(['PassengerId'],axis=1)
x.info

#check if missing values

x.isnull().values.any()

x[pd.isnull(x).any(axis=1)]

x.Age.fillna(x.Age.mean(),inplace=True)
x.isnull().values.any()
print(x.head())

#split dataset into traning and validation

from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=.25,random_state=7)

#get a baseline

def simple_heuristic(titanicdf):
  predictons=[]
  for passenger_index,passenger in titanicdf.iterrows():
    if passenger['Sex']==1:
      predictons.append(1)
    elif passenger['Age']<18 and passenger['Pclass']==1:
      predictons.append(1)
    else:
      predictons.append(0)
  return predictons

simplePredictions=simple_heuristic(x_valid)
correct= sum(simplePredictions== y_valid)
print('baseline:',correct/len(y_valid))

#logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()

model.fit(x_train,y_train)

print("score of train :",model.score(x_train,y_train))

print("score of vaild:",model.score(x_valid,y_valid))

print(model.slope)

print(model.coef_)

model.intercept_

titanic.corr()