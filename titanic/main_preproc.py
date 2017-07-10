import pandas as pd
#from pandas import Series,DataFrame
import sklearn.preprocessing as preprocessing

def one_hot_scaler(df):

    dummies_cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
    dummies_embarked = pd.get_dummies(df['Embarked'],prefix='Embarked')
    dummies_sex = pd.get_dummies(df['Sex'],prefix='Sex')
    dummies_pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')

    df = pd.concat([df,dummies_cabin,dummies_embarked,dummies_pclass,dummies_sex],axis=1)
    df.drop(['Cabin','Embarked','Pclass','Sex','Ticket','Name'],axis=1,inplace=True)


    scaler = preprocessing.StandardScaler()
    df['Age_scaled']=df['Age']
    df['Fare_scaled']=df['Fare']
    df[['Age_scaled','Fare_scaled']] = scaler.fit_transform(df[['Age','Fare']]) # double []

    return df



