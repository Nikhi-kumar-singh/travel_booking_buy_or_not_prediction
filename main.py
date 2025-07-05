from import_file import *
from input_data_file import input_data
from data_cleaning_file import data_cleaning
from data_transformation_file import data_transformation
from model_tuner_file import model_tuner
from test_model_file import test_model
from content.select_classification_model_file import select_model
from content.select_content_file import select_content


file_name,output,regression=select_content()

df=input_data(file_name)

df=data_cleaning(df)

x_train_scaled,x_test_scaled,y_train,y_test=data_transformation(df,output)

model_name,params=select_model(2)

# this is updated form here
tuner=RandomizedSearchCV

model,score=model_tuner(model_name,params,x_train_scaled,y_train,tuner)

test_model(model,x_test_scaled,y_test,regression)

print(f"best score for {model_name} : {score}")