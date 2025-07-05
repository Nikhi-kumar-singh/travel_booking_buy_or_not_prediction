from import_file import *



def data_transformation(df,output,num_scaler=StandardScaler(),cat_scaler=OneHotEncoder(drop="first",handle_unknown="ignore")):
  num_features=df.select_dtypes(exclude="O").columns
  cat_features=df.select_dtypes(include="O").columns

  num_features=[num for num in num_features if num!=output]

  x=df.drop([output],axis=1)
  y=pd.DataFrame(df[output],columns=[output])
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43)

  scaler=ColumnTransformer([
      ("num_feature_scaler",num_scaler,num_features),
      ("cat_feature_scaler",cat_scaler,cat_features)
  ])

  scaler.fit_transform(x_train)
  x_train_scaled=scaler.transform(x_train)
  x_test_scaled=scaler.transform(x_test)

  return x_train_scaled,x_test_scaled,y_train,y_test