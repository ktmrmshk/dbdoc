# Databricks notebook source
# MAGIC %md #製造におけるIoTデータ分析 on Databricks
# MAGIC 
# MAGIC 
# MAGIC ## Part 3 - 機械学習
# MAGIC 
# MAGIC このノートブックでは、Azure上でIIoTのIngest、Processing、Analyticsを行うための以下のアーキテクチャをデモしています。デモでは以下のようなアーキテクチャが実装されています。
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_architecture.png" width=800>
# MAGIC 
# MAGIC ノートブックは以下の手順でセクションに分かれています。
# MAGIC 3. **機械学習** - 分散型MLを用いてXGBoost回帰モデルを学習し、過去のセンサーデータから電力出力と資産の残り寿命を予測する。
# MAGIC 4. **モデルのデプロイ** - 学習したモデルをDatabricksのモデルレジストリにデプロイし、リアルタイムに提供する。
# MAGIC 5. **モデル推論** - ホストされたモデルに対して、REST APIを介して実データを即座にスコアリングします。

# COMMAND ----------

# MAGIC %md
# MAGIC The Lakehouse architecture pattern allows the *best tool for the job* to be brought *to the data*. We have performed data engineer, SQL analytics, and now data science & ML - **all on the same data living in our organization's Data Lake**. Delta + ADLS provides security, reliability and performance to all data sets - stream, batch, structured, unstructured - and opens the data up to any analytics workload. 
# MAGIC 
# MAGIC Lakehouseのアーキテクチャパターンでは、「仕事に最適なツール」を「データ」に近づけることができます。我々は、データエンジニア、SQLアナリティクス、そして今ではデータサイエンスとMLを実行しています。Delta + ADLSは、ストリーム、バッチ、構造化、非構造化など、すべてのデータセットにセキュリティ、信頼性、パフォーマンスを提供し、あらゆる分析ワークロードにデータを開放します。
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Lakehouse.png" width=800>

# COMMAND ----------

# モデルを展開するためのAzureMLワークスペース情報（名前、リージョン、リソースグループ、サブスクリプションID)
dbutils.widgets.text("Storage Account", "<your storage account>", "Storage Account")

# COMMAND ----------

# MAGIC %md ## Step 1 - 環境のセットアップ
# MAGIC 
# MAGIC 前提条件は以下の通りです。
# MAGIC 
# MAGIC ### 必要なAzureサービス
# MAGIC * ADLS Gen 2 Storage account (コンテナ名: `iot`)
# MAGIC 
# MAGIC ### Azure Databricksの設定
# MAGIC * 3ノード(最小)のDatabricks Cluster(Runtime: DBR 7.0以上)、かつ以下のライブラリ:
# MAGIC * 以下のSecrets(scope名:`iot`)
# MAGIC  * `adls_key` - ADLS storage accountへのアクセスキー **(重要 - [Access Key](https://raw.githubusercontent.com/tomatoTomahto/azure_databricks_iot/master/bricks.com/blog/2020/03/27/data-exfiltration-protection-with-azure-databricks.html))を使用すること**
# MAGIC 
# MAGIC **注意** Part1のNotebookで作成したデータセットを使用します。

# COMMAND ----------

# 一時データ用のストレージアカウントへのアクセスを設定(Synapseへのプッシュ時に使用)
storage_account = dbutils.widgets.get("Storage Account")
spark.conf.set(f"fs.azure.account.key.{storage_account}.dfs.core.windows.net", dbutils.secrets.get("iot","adls_key"))

# ストレージ上のパスを設定する
ROOT_PATH = f"abfss://iot@{storage_account}.dfs.core.windows.net/manufacturing_demo/"

# Pythonライブラリのimport
import os, json, requests
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
import numpy as np 
import pandas as pd
import xgboost as xgb
import mlflow.xgboost

# COMMAND ----------

# MAGIC %md ## Step 3 - 機械学習
# MAGIC 
# MAGIC センサーデバイスからデータレイク・ストレージのエンリッチド・デルタ・テーブルに確実にデータが流れている今、メンテナンスイベントがいつ発生するかを予測するMLモデルの構築に着手することができます。そのためには、将来のセンサー値を予測し、その値が設備の安全動作範囲を逸脱したときに特定します。
# MAGIC 
# MAGIC ***製造現場ごとに***温度予測モデルを作成していきます。
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/manufacturing_pred_maint.png" width=800>
# MAGIC 
# MAGIC 回帰モデルの学習にはXGBoostフレームワークを使用します。データの大きさとFacilityの数を考慮して、Spark UDFを使用してクラスタ内のすべてのノードにトレーニングを分散します。

# COMMAND ----------

# MAGIC %md ### 3a. 特徴量エンジニアリング
# MAGIC 
# MAGIC 5分先の気温を予測するためには、まずデータをタイムシフトしてラベルカラムを作成する必要があります。この作業はSpark Window Partitioningを使って簡単に行うことができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE VIEW manufacturing.ml_feature_view AS
# MAGIC SELECT facilityid, temperature, humidity, pressure, moisture, oxygen, radiation, conductivity,
# MAGIC   LEAD(temperature, 1, temperature) OVER (PARTITION BY facilityid ORDER BY window) as next_temperature
# MAGIC FROM manufacturing.sensors_enriched;
# MAGIC 
# MAGIC SELECT * FROM manufacturing.ml_feature_view

# COMMAND ----------

# MAGIC %md ### 3b. 分散型モデルトレーニング - 出力の予測
# MAGIC [Pandas UDFs](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/udf-python-pandas?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json)により、Pandasのコードをクラスタ内の複数のノードに渡ってベクトル化することができます。ここでは、特定の製造施設のすべての履歴データに対してXGBoost RegressorモデルをトレーニングするUDFを作成します。施設グループレベルでモデルトレーニングを行うため、Grouped Map UDFを使用します。

# COMMAND ----------

# 施設のデータについて、XGBoost Regressorを使って学習させる関数を作成
def train_distributed_xgb(readings_pd, label_col, prediction_col):
  mlflow.xgboost.autolog()
  with mlflow.start_run():
    # モデルタイプとファシリティIDのログ
    mlflow.log_param('facilityid', readings_pd['facilityid'][0])

    #  この施設のデータでXGBRegressorをトレーニングします。
    alg = xgb.XGBRegressor() 
    train_dmatrix = xgb.DMatrix(data=readings_pd[feature_cols].astype('float'),label=readings_pd[label_col])
    params = {'learning_rate': 0.5, 'alpha':2, 'colsample_bytree': 0.5, 'max_depth': 5}
    model = xgb.train(params=params, dtrain=train_dmatrix, evals=[(train_dmatrix, 'train')])

    # データセットを予測し、その結果を返す
    readings_pd[prediction_col] = model.predict(train_dmatrix)
  return readings_pd

# Read in our feature table and select the columns of interest
feature_df = spark.table('manufacturing.ml_feature_view').selectExpr('facilityid','temperature', 'humidity', 'pressure', 'moisture', 'oxygen', 'radiation', 'conductivity','next_temperature','0 as next_temperature_predicted')

# XGBモデルの学習をSparkで配信するためのPandas UDFの登録
@pandas_udf(feature_df.schema, PandasUDFType.GROUPED_MAP)
def train_temperature_model(readings_pd):
  return train_distributed_xgb(readings_pd, 'next_temperature', 'next_temperature_predicted')

# COMMAND ----------

# 特徴的なデータセットに対してPandas UDFを実行します。
temperature_predictions = (
  feature_df.groupBy('facilityid')
    .apply(train_temperature_model)
    .write.format("delta").mode("overwrite")
    .option("path",ROOT_PATH + "gold/temperature_predictions")
    .saveAsTable("manufacturing.temperature_predictions")
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- 実測値と予測値の比較
# MAGIC SELECT * FROM manufacturing.temperature_predictions

# COMMAND ----------

# MAGIC %md #### Databricksにおけるモデルトラッキングの自動化
# MAGIC 
# MAGIC モデルをトレーニングする際、Databricksが管理するMLflowがノートブックの「Runs」タブで各ランを自動的に追跡する様子に注目してください。各ランを開き、MLflow Autologgingによって取得されたパラメータ、メトリクス、モデル、モデルの成果物を確認することができます。XGBoost Regression モデルの場合、MLflow は以下を追跡します。
# MAGIC 
# MAGIC 1. 1. `params` 変数に渡されたモデルパラメータ（alpha、colsample、learning rate など） 2.
# MAGIC 2. evals`で指定されたメトリクス（デフォルトではRMSE
# MAGIC 3. 学習されたXGBoostモデルファイル
# MAGIC 4. 特徴のインポート
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iiot_mlflow_tracking.gif" width=800>

# COMMAND ----------

# MAGIC %md 
# MAGIC 残りの温度を予測するモデルは、MLflowによって学習され、記録されました。これでDatabricksにモデルを展開する作業に移ることができます。

# COMMAND ----------

# MAGIC %md ## Step 4 - MLflowへのモデル展開
# MAGIC 
# MAGIC モデルがトレーニングされたので、自動化された方法でAzure MLやMLflowのようなモデル提供環境に直接デプロイすることができます。以下では、デプロイメントのトラッキングとサービングのために、最もパフォーマンスの高いモデルをDatabricks-nativeホストのMLflowモデルレジストリに登録します。登録が完了したら、サービングを有効にしてモデルにREST APIを公開します。モデルの登録は、次のセルに示すように、UIまたはMLflow APIを使って行うことができます。

# COMMAND ----------

# デプロイする設備を指定する
facility = "FAC-0"

# 最良のパフォーマンスのモデル（最小RMSE）を取得する
best_model = mlflow.search_runs(filter_string=f'params.facilityid="{facility}"')\
  .dropna().sort_values("metrics.train-rmse")['artifact_uri'].iloc[0] + '/model'

# 最も性能の良いモデルをDatabricksのモデルレジストリに登録する
mlflow.register_model(best_model, "temperature_prediction")

# COMMAND ----------

# MAGIC %md ## Step 5 - モデル推論: リアルタイム・スコアリング
# MAGIC 
# MAGIC モデルが登録されると、以下のようにモデル・レジストリでモデルを追跡し、モデル・サービングを有効にすることができます。
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/mlflow_register_serve.gif" width=800>
# MAGIC 
# MAGIC Webアプリ、PowerBI、またはDatabricksからホストされたモデルのURIに直接HTTP RESTコールを行い、データを直接取得できるようになりました。

# COMMAND ----------

import os
import requests
import pandas as pd

# スコアリングのためのペイロードの構築（例：現在のセンサーの状態から5分後の温度を得る)
payload = [{
  'temperature':25.4,
  'humidity':67.2, 
  'pressure':33.6, 
  'moisture':49.8, 
  'oxygen':27.7, 
  'radiation':116.3, 
  'conductivity':128.0
}]

# APIを呼び出してデータを取得
temp_prediction_uri = "https://adb-5016390217096892.12.azuredatabricks.net/model/temperature_prediction/1/invocations" # MLflowやAzureMLのモデルURIに置き換えてください。
databricks_token = "????" # あなたのDatabricks Personal Access Tokenに置き換えてください。

# RESTを使ってAPIを呼び出し、結果を返す機能
def score_model(uri, payload):
  headers = {'Authorization': f'Bearer {databricks_token}'}
  data_json = payload
  response = requests.request(method='POST', headers=headers, url=uri, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

prediction = score_model(temp_prediction_uri, payload)

print(f'Temperature predicted from model: {int(prediction[0])}°C')

# COMMAND ----------

# MAGIC %md ### Step 6: 予知保全
# MAGIC 
# MAGIC これで、温度が安全動作しきい値を上回ったり下回ったりしたときに故障につながる動作条件を特定することができます。用意した温度モデルを繰り返し呼び出して感度分析を行うことで、故障やメンテナンスイベントにつながる圧力と温度の組み合わせのヒートマップを作成することができます。

# COMMAND ----------

min_temp = 25.0
max_temp = 27.0

# リクエストと一緒に送信するペイロードを構築する
payload = [{
  'temperature':25.4,
  'humidity':67.2, 
  'pressure':33.6, 
  'moisture':49.8, 
  'oxygen':27.7, 
  'radiation':116.3, 
  'conductivity':128.0
}]

from numpy import arange

# 50種類の回転数構成を繰り返し、各回転数での予測電力と残存寿命を把握する。
results = []
for temperature in arange(24.0,26.0,0.2):
  for pressure in arange(34.0,36.0,0.2):
    payload[0]['temperature'] = temperature
    payload[0]['pressure'] = pressure
    expected_temperature = score_model(temp_prediction_uri, payload)[0]
    failure = 0 if expected_temperature < max_temp else 1
    results.append((temperature, pressure, expected_temperature, failure))
  
# 各RPM構成で発生する収益、コスト、利益を計算する
matrix_df = pd.DataFrame(results, columns=['Temperature', 'Pressure', 'Expected Temperature', 'Failure'])

display(matrix_df)

# COMMAND ----------

# MAGIC %md 温度が25.2～25.6、圧力が34～35.4の場合、5分後には温度が安全動作閾値を外れて故障していることがわかります。

# COMMAND ----------

# MAGIC %md ### (補足) - Azure MLへのデプロイ
# MAGIC 
# MAGIC 多くのお客様は、AzureMLのような外部の中央MLホスティングツールを使用して、Databricksで開発されたものであれ、外部（ラップトップ、AzureML、AutoMLフレームワークなど）で開発されたものであれ、すべてのモデルを提供したいと考えるでしょう。Databricksでトレーニングされたモデルは、クラスタにインストールされた[mlflow-azureml]ライブラリを使って、簡単にAzureMLにデプロイできます。IoTモデルをAzureMLにデプロイして提供するコードスニペットについては、サンプルノートブック[こちら](https://databricks.com/notebooks/iiot/iiot-end-to-end-part-2.html)をご覧ください。
