import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel("titanic.xlsx")
# print(df.head(10))

def sex_encoder(data):
    mapper={"male":0,"female":1}  
    data["Sex"]=data["Sex"].replace(mapper) 
    return data

df = sex_encoder(df)

def embarked_encoder(data):
    df=pd.get_dummies(data=data["Embarked"],prefix='Embarked')
    data=pd.concat([data,df],axis=1) 
    data.drop(["Embarked"],axis=1,inplace=True)
    return data
    
df = embarked_encoder(df)

# print(df.columns)
X = df[['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = df[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred=lr.predict(X_test)
lr_proba=lr.predict_proba(X_test) 
accuracy = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {accuracy}")



# ## Saving Model

joblib.dump(lr, "titaniv_v0.sav")

