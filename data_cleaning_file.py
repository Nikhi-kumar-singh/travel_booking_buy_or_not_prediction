from import_file import *


def data_cleaning(df):
  num_features=df.select_dtypes(exclude="O").columns
  cat_features=df.select_dtypes(include="O").columns

  for column in df.columns:
    if column in cat_features:
      df[column]=df[column].fillna(df[column].mode()[0])
    else:
      df[column]=df[column].fillna(df[column].mean())

  return df