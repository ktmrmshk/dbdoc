# Databricks notebook source
import mlflow

mlflow.start_run()
mlflow.log_param('project_name', 'foo')
mlflow.log_metric('score', 123.45)
mlflow.end_run()

# COMMAND ----------

import mlflow
with mlflow.start_run() as run:
  mlflow.log_params({'name': 'masa', 'age': 123})

# COMMAND ----------

run = mlflow.active_run()
print(run)

# COMMAND ----------

mlflow.start_run()
run = mlflow.active_run()
print(run)

# COMMAND ----------

mlflow.log_metric('score', 3.14)
mlflow.log_metrics({'age': 32, 'step': 4.12})

# COMMAND ----------

run.info


# COMMAND ----------

d={'foo': 123.122, 'name': 'masa', 'info': [1, 2, 'age', {'birth': 'satte'}]}

# COMMAND ----------

mlflow.log_dict(d, 'dict.json')
mlflow.log_dict(d, 'dir/data.yaml')

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0, 1], [2, 3])

# COMMAND ----------

mlflow.log_figure(fig, 'fig.png')

# COMMAND ----------

from PIL import Image

img = Image.new('RGB', (100,100))

# COMMAND ----------

mlflow.log_image(img, 'sample.png')

# COMMAND ----------

mlflow.log_text('this is a pen', 'penpen.txt')

# COMMAND ----------

mlflow.set_tag('tagname123', 'tagvalue456')

# COMMAND ----------

# MAGIC %sh ls logs/2021-09-18-13.log.gz

# COMMAND ----------

mlflow.log_artifact('logs/2021-09-18-13.log.gz')

# COMMAND ----------


