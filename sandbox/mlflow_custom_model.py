# Databricks notebook source
import mlflow.pyfunc
import mlflow

class AddN(mlflow.pyfunc.PythonModel):
  def __init__(self, n):
    self.n = n
  def predict(self, context, model_input):
    return model_input.apply(lambda column: column + self.n)

with mlflow.start_run():
  model_path = "add_n_model"
  add5_model = AddN(5)
  mlflow.pyfunc.log_model('model_path', python_model=add5_model)

# COMMAND ----------


