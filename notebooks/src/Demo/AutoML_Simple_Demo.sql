-- Databricks notebook source
-- MAGIC %md
-- MAGIC 
-- MAGIC # Databricks AutoMLデモ
-- MAGIC 
-- MAGIC **環境設定**
-- MAGIC 
-- MAGIC * `Databricks Runtime v8.3 +ML` 以上のクラスタを使用してください。
-- MAGIC 
-- MAGIC ### 概要
-- MAGIC 
-- MAGIC 1. データの準備・EDA
-- MAGIC 1. AutoMLにより学習モデルを作成
-- MAGIC 3. モデル学習結果の確認
-- MAGIC 1. モデルのデプロイと推論の実施

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. データの準備・EDA

-- COMMAND ----------

-- MAGIC %fs ls /databricks-datasets/wine-quality/

-- COMMAND ----------

-- DBTITLE 1,サンプルデータ(CSV)の確認
-- MAGIC %fs head dbfs:/databricks-datasets/wine-quality/winequality-white.csv

-- COMMAND ----------

-- DBTITLE 1,環境のクリーンアップ1(ファイル)
-- MAGIC %fs rm -r /tmp/YOURNAME_automl/

-- COMMAND ----------

-- DBTITLE 1,環境のクリーンアップ2(DATABASE)
DROP DATABASE IF EXISTS  YOURNAME_automl CASCADE;
CREATE DATABASE YOURNAME_automl;
USE YOURNAME_automl;

-- COMMAND ----------

-- DBTITLE 1,CSVの読み込み
CREATE TABLE wine_quality_raw
(
  fixed_acidity DOUBLE,
  volatile_acidity DOUBLE,
  citric_acid DOUBLE,
  residual_sugar DOUBLE,
  chlorides DOUBLE,
  free_sulfur_dioxide DOUBLE,
  total_sulfur_dioxide DOUBLE,
  density DOUBLE,
  pH DOUBLE,
  sulphates DOUBLE,
  alcohol DOUBLE,
  quality INT
)
USING CSV
OPTIONS (
  sep = ';',
  header = 'true',
  inferSchema = 'true'
)
LOCATION 'dbfs:/databricks-datasets/wine-quality/winequality-white.csv';

SELECT * FROM wine_quality_raw

-- COMMAND ----------

-- DBTITLE 1,Delta Tableに書き込む(Delta Table化)
CREATE TABLE wine_quality_delta
USING delta
LOCATION '/tmp/YOURNAME_automl/wine_quality_delta'
AS (
  SELECT * FROM wine_quality_raw
);

SELECT * FROM wine_quality_delta;

-- COMMAND ----------

-- DBTITLE 1,データの可視化・EDA
SELECT * FROM wine_quality_delta;

-- COMMAND ----------

DESCRIBE EXTENDED wine_quality_delta;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ## 2. AutoMLにより学習モデルを作成
-- MAGIC 
-- MAGIC ワインの品質を推定するモデルをAutoMLを使用して作成する。
-- MAGIC 
-- MAGIC * 説明変数(特徴量): `quality`以外のフィールド
-- MAGIC * 目的変数(推定のターゲット): `quality`フィールド
-- MAGIC 
-- MAGIC 
-- MAGIC 左メニューから > Machine Learning > Experiments > Craete AutoML Expriments を選択。
-- MAGIC 
-- MAGIC 以下の通り設定してください。
-- MAGIC (ここではデモのための実行サイズを小さくしたパラメータを選択しています)
-- MAGIC 
-- MAGIC * `Compute`: `Databricks Runtime v8.3 +ML` 以上のクラスタを選択。起動済みである必要があります。
-- MAGIC * `ML problem type`: `Regression` (回帰)
-- MAGIC * `Dataset`: 上記で準備した`YOURNAME_automl` > `wine_quality_delta`　を選択
-- MAGIC * `Prediction target`: `quality`
-- MAGIC * (Adavnced Option > `Stopping Conditions`) 
-- MAGIC   - `Timeout`: `10 min`
-- MAGIC   - Number of trial runs: `20`
-- MAGIC 
-- MAGIC 最後に`Start AutoML`をクリックする。
-- MAGIC 
-- MAGIC 
-- MAGIC ![automl_start](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/automl/automl_setup_lite.gif)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ##3. モデル学習結果の確認
-- MAGIC 
-- MAGIC AutoMLの実験結果ページから確認できます。
-- MAGIC 
-- MAGIC ### EDA Notebook
-- MAGIC 
-- MAGIC ![EDA_notebook](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/automl/EDA_notebook.gif)
-- MAGIC 
-- MAGIC 
-- MAGIC ### 最適なモデル
-- MAGIC 
-- MAGIC ![model_selection](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/automl/model_selection.gif)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. モデルのデプロイと推論の実施
-- MAGIC 
-- MAGIC **注意**
-- MAGIC 今回はサンプルデータの都合により、使用方法、動作確認を目的として、学習に使用したデータに対してモデル推論を実施します。

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC import mlflow
-- MAGIC 
-- MAGIC # ロードするモデルの指定
-- MAGIC # runs:/xxxxx部分を置き換えてください
-- MAGIC logged_model = 'runs:/4bb9a56525f043b08925c8acd7743e23/model'
-- MAGIC 
-- MAGIC # モデルをロードする
-- MAGIC loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC df = sql('SELECT * FROM wine_quality_delta')
-- MAGIC 
-- MAGIC # 特徴量だけのカラム名を抜き出す
-- MAGIC column_names = df.drop('quality').columns
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # 推論を実行
-- MAGIC pred_df = df.withColumn('predictions', loaded_model(*column_names)).collect()
-- MAGIC 
-- MAGIC # 確認
-- MAGIC display(pred_df)
