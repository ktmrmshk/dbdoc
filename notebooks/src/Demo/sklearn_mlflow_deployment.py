# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # MLflowによる機械学習のワークフロー管理
# MAGIC ## (学習からデプロイまで)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Scikit-learnを使った学習をMLflowのauto_logで自動トラックする

# COMMAND ----------

import os, warnings, sys, logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse


def eval_metrics(actual, pred):
    rmse = np.sqrt( mean_squared_error(actual, pred) )
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 学習データの読み込み

# COMMAND ----------

np.random.seed(40)

csv_url = ('/dbfs/databricks-datasets/wine-quality/winequality-red.csv')

data = pd.read_csv(csv_url, sep=';')
        
train, test = train_test_split(data)

train_x = train.drop(['quality'], axis=1)
test_x = test.drop(['quality'], axis=1)

train_y = train[ ['quality'] ]
test_y = test[['quality']]


# COMMAND ----------

test_x.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### データ可視化・探索 (EDA)
# MAGIC 
# MAGIC `display()`を使うと効率よくデータ可視化・探索(EDA)ができます。
# MAGIC 
# MAGIC 必要であれば前処理などもそのまま実行できます。

# COMMAND ----------

display(data)

# COMMAND ----------

# いろいろプロットを試してみてください

display(data)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### モデルの構築(トレーニング)

# COMMAND ----------

import mlflow, mlflow.sklearn
mlflow.sklearn.autolog()

alpha = 0.02
l1_ratio = 0.01

with mlflow.start_run():
  lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
  lr.fit(train_x, train_y)

  pred = lr.predict(test_x)

  (rmse, mae, r2) = eval_metrics(test_y, pred)

  print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. MLflowからモデルをロードする
# MAGIC 
# MAGIC Databricks上でのモデルのデプロイは以下の3通りに分けられます。
# MAGIC 
# MAGIC 1. バッチ処理: Databricks上のnotebookでDataframeを入力し、スコアリングするコードを定期実行する
# MAGIC 1. ストリーミング処理: Databricks上のnotebookでストリーミングDataframeを入力し、スコアリングを逐次実行する
# MAGIC 1. REST Serving: REST Server上にモデルをデプロイし、HTTPリクエストでスコアリングデータを読み込み、レスポンスで推定結果を返す
# MAGIC 
# MAGIC Databricks上ではバッチ処理、ストリーミング処理がDataframe的に同等に扱えるため、上記のバッチ処理、ストリーミング処理はほぼ同じデプロイ方法になります。
# MAGIC Rest Servingについては、MLflowのレジストリUIからデプロイ可能です。
# MAGIC 
# MAGIC ここでは、バッチ処理、ストリーミング処理でデプロイする方法を見ていきます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC MLflowからモデルをロードする場合、以下の2通りがあります。
# MAGIC 
# MAGIC 1. Run(学習の結果)のIDを指定してロードする
# MAGIC 1. モデルレジストリから、モデル名、version(もしくは、staging, production)を指定してロードする
# MAGIC 
# MAGIC さらに、ロードしたモデルは、以下の2通りの種類でロードできます。
# MAGIC 
# MAGIC 1. SparkのDataframeを入力できるPythonの関数としてロード
# MAGIC 1. PandasDataframeを入力できるPythonの関数としてロード

# COMMAND ----------

# MAGIC %md
# MAGIC ### A. Run IDからSpark Dataframeのpython関数としてデプロイする

# COMMAND ----------

import mlflow

# 実際のRun IDで置き換えてください!
logged_model = 'runs:/3b1aeb32524244b98006da4c7a2b7211/model'

# モデルをロードする
model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
model

# COMMAND ----------

# 推定を実施する(スコアリングを実施する)対象のデータを読み込む

df = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .option('sep', ';')
  .load('/databricks-datasets/wine-quality/winequality-red.csv')
)

display(df)

# COMMAND ----------

# モデルを適用して推定する(スコアリング)
pred_df = df.withColumn('pred', model(*df.columns))
display(pred_df)

# COMMAND ----------

# MAGIC %md ### B. Run IDからPandas Dataframeのpython関数としてデプロイする

# COMMAND ----------

import mlflow

# 実際のRun IDで置き換えてください!
logged_model = 'runs:/3b1aeb32524244b98006da4c7a2b7211/model'

pd_model = mlflow.pyfunc.load_model(logged_model)
pd_model

# COMMAND ----------

# 推定を実施する(スコアリングを実施する)対象のデータを読み込む
# (先ほど読み込んだSpark DataframeをPandas Dataframeに変換する)

pd_df = df.toPandas()
pd_df

# COMMAND ----------

# モデルを適用して推定する(スコアリング)
import pandas as pd
pred_array = pd_model.predict(pd_df)

# COMMAND ----------

# 結果をDataframeにまとめて、表示
pd_df['pred'] = pred_array
pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### C. モデルレジストリから、モデル名、version(もしくは、staging, production)を指定してロードする

# COMMAND ----------

import mlflow.pyfunc

model_name = "レジストリ上のモデルの名前"
model_version = 1
# model_version = 'production' ## <= このようにproduction/stagingも指定可能

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
