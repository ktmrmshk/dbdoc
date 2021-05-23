# Databricks notebook source
# MAGIC %md
# MAGIC ### Task001のnotebook
# MAGIC 
# MAGIC * workflowをまとめるnotebook(`master_notebook`)から呼び出される想定
# MAGIC * 処理を実行して、最後に`Job Done by Task001`という文字列を返す

# COMMAND ----------

# 適当な処理
print('Hello! by Task001')

# COMMAND ----------

# 戻り値を設定できる
dbutils.notebook.exit("Job Done by Task001")
