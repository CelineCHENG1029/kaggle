from sklearn import linear_model, preprocessing, model_selection
import main_preproc
import pandas as pd
import age_proc_rf

data_train_raw = pd.read_csv("data/train.csv")
data_train,rfr = age_proc_rf.set_missing_ages(data_train_raw)
data_train = age_proc_rf.set_cabin_type(data_train)
df = main_preproc.one_hot_scaler(data_train)

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
y = train_np[:,0]
X = train_np[:,1:]
clf = linear_model.LogisticRegression(C=0.1,penalty='l1',tol=1e-6)
clf.fit(X,y)

coef_table = pd.DataFrame({'Columns':list(train_df.columns)[1:], 'Coef':list(clf.coef_.T)})

cross_validation_scores = model_selection.cross_val_score(clf,X,y,cv=5)

split_data_train_raw,split_data_validation_raw = model_selection.train_test_split(data_train_raw,test_size=0.3,random_state=0)
split_data_train_np,split_data_validation_np = model_selection.train_test_split(train_np,test_size=0.3,random_state=0)
split_train_y = split_data_train_np[:,0]
split_train_x = split_data_train_np[:,1:]
clf_split = linear_model.LogisticRegression(C=0.5,penalty='l1',tol=1e-6)
clf_split.fit(split_train_x,split_train_y)

cross_validation_prediction = clf_split.predict(split_data_validation_np[:,1:])
bad_cases = data_train_raw.loc[data_train_raw['PassengerId'].isin \
    (split_data_validation_raw[cross_validation_prediction != split_data_validation_np[:,0]]['PassengerId'].values)]
