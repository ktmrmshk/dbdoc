# Databricks notebook source
# MAGIC %md
# MAGIC ### Task002のnotebook
# MAGIC 
# MAGIC * workflowをまとめるnotebook(`master_notebook`)から呼び出される想定
# MAGIC * パラメータ`name`, `age`を呼び出し元から受け取る
# MAGIC * 処理を実行して、最後に`Job Done by Task002`という文字列を返す

# COMMAND ----------

# パラメータ`name`, `age`を呼び出し元から受け取る

dbutils.widgets.text("name", "default")
dbutils.widgets.text("age", "0")

name = dbutils.widgets.get("name").strip()
age = int( dbutils.widgets.get("age").strip() )

print(f'name => {name}')
print(f'age => {age}')

# COMMAND ----------

# 適当な処理
print('Hello! by Task002')

# COMMAND ----------

# 戻り値を設定できる
dbutils.notebook.exit("Job Done by Task002")
