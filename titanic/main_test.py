import pandas as pd
import numpy as np

import main_train as mt
import age_proc_rf
import main_preproc
import main_train

data_test = pd.read_csv("data/test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare']=0
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

X = null_age[:,1:]
predicted_age = mt.rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predicted_age

data_test = age_proc_rf.set_cabin_type(data_test)
df_test = main_preproc.one_hot_scaler(data_test)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
prediction = main_train.clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':prediction.astype(np.int32)})

result.to_csv("data/prediction.csv",index=False)