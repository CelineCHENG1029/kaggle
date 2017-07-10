from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):

    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    x = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)

    predictAge = rfr.predict(unknown_age[:,1:])
    df.loc[(df.Age.isnull()),'Age']=predictAge

    return  df,rfr

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df