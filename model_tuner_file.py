from  import_file import *



def model_tuner(model_name,params,x_train,y_train,tuner=RandomizedSearchCV):
  tuner_model=tuner(
      estimator=model_name(),
      param_distributions=params,
      scoring="accuracy",
      cv=5,
      n_jobs=-1,
      verbose=2
  )

  tuner_model.fit(x_train,y_train)

  return tuner_model.best_estimator_,tuner_model.best_score_