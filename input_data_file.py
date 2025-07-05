from  import_file import *


def input_data(file_name):
  df=pd.read_csv(file_name)
  return df