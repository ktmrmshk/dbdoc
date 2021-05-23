# Databricks notebook source
# MAGIC %md
# MAGIC # NotebookのWidget機能
# MAGIC 
# MAGIC notebookのWidget機能で簡易的なUIが実装可能
# MAGIC 
# MAGIC 詳細は[document](https://docs.microsoft.com/ja-jp/azure/databricks/notebooks/widgets)を参照ください。

# COMMAND ----------

# notebookのWidget機能で簡易的なUIが実装可能

# textのwidget
dbutils.widgets.text("widget_text123", "defaultValue1")

# Dropdownのwidget
dbutils.widgets.dropdown("MyWidget", "one", ["one", "two", "three", "million!"])

# combobox
dbutils.widgets.combobox("product_combo", "Other", ["Camera", "GPS", "Smartphone"])

# multiselect
dbutils.widgets.multiselect("product_multiselect", "Camera", ["Camera", "GPS", "Smartphone"])

# COMMAND ----------

# それぞれのwidgetの値を取得する

print( dbutils.widgets.get('widget_text123') )
print( dbutils.widgets.get('MyWidget') )
print( dbutils.widgets.get('product_combo') )
print( dbutils.widgets.get('product_multiselect') )

# COMMAND ----------

# widgetをずべて削除する
dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md # Notebookのワークフロー管理
# MAGIC 
# MAGIC 外部のnotebookを呼ぶ機能を用いて、このnotebookから外部のnotebookを呼び出す。
# MAGIC 
# MAGIC Notebookのディレクトリ構成
# MAGIC 
# MAGIC * `master_notebook` <= このnotebook
# MAGIC * `notebooks/`
# MAGIC   - `task001` <= 実際の処理コードが書かれているnotebook
# MAGIC   - `task002` <= 実際の処理コードが書かれているnotebook

# COMMAND ----------

# notebook "task001"を実行する

# timeout=60sec
ret = dbutils.notebook.run("./notebooks/task001", 60)
print(ret)

# 結果のリンクからnotebookの実行結果の詳細を参照できます。

# COMMAND ----------

# notebook "task002"を実行する

# パラメータ name=abc123, age=32 をnotebookに渡す
#  => 渡されたnotebook側では、widgetとして認識されるので、dbutils.widget.get()を使用してこのパラメータを取得する
ret = dbutils.notebook.run("./notebooks/task002", 60, {"name": "abc123", "age": "23"})
print(ret)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## エラー時のリトライ制御
# MAGIC 
# MAGIC 基本的にはPythonコードでリトライを実装する。
# MAGIC 詳細は[こちら](https://docs.microsoft.com/ja-jp/azure/databricks/notebooks/notebook-workflows#python)を参照ください。

# COMMAND ----------

#以下の関数が dbutils.notebook.runのwrapper関数でリトライ機能を提供する
def run_with_retry(notebook, timeout, args = {}, max_retries = 3):
  num_retries = 0
  while True:
    try:
      return dbutils.notebook.run(notebook, timeout, args)
    except Exception as e:
      if num_retries > max_retries:
        raise e
      else:
        print("Retrying error", e)
        num_retries += 1

run_with_retry("./notebooks/task001", 60, max_retries = 5)

# COMMAND ----------

# MAGIC %md
# MAGIC # 参考: Databricksの通知機能の実装について
# MAGIC 
# MAGIC Pythonのrequestsライブラリを使用して、Webhookとの連携が可能です。これにより、Slackをはじめとするメッセージングサービスとも連携が可能です。
# MAGIC 
# MAGIC サンプルコード:
# MAGIC ```
# MAGIC def postToAPIEndpoint(content, webhook=""):
# MAGIC     """
# MAGIC     Post message to Teams to log progress
# MAGIC     """
# MAGIC     import requests
# MAGIC     from requests.exceptions import MissingSchema
# MAGIC     from string import Template
# MAGIC 
# MAGIC     t = Template('{"text": "${content}"}')
# MAGIC 
# MAGIC     try:
# MAGIC         response = requests.post(
# MAGIC             webhook,
# MAGIC             data=t.substitute(content=content),
# MAGIC             headers={"Content-Type": "application/json"},
# MAGIC         )
# MAGIC         return response
# MAGIC     except MissingSchema:
# MAGIC         print(
# MAGIC             "Please define an appropriate API endpoint use by defining the `webhook` argument"
# MAGIC         )
# MAGIC 
# MAGIC 
# MAGIC postToAPIEndpoint("This is my post from Python", webhookMLProductionAPIDemo)
# MAGIC ```

# COMMAND ----------


