# Databricks notebook source
# MAGIC %md
# MAGIC # サンプルのCSVファイルを読み込み、Deltaテーブルを作成する
# MAGIC 
# MAGIC ### 概要
# MAGIC 1. CSVファイルをSparkのDataFrameとして読み込む
# MAGIC 2. DataFrameをDeltaフォーマットとして保存する
# MAGIC 3. 続いて、SQLから参照できるように、Deltaテーブルを作成する(上記のDeltaファイルと紐付ける)
# MAGIC 4. Deltaテーブルを参照する(プロットなどを試す)

# COMMAND ----------

# DBTITLE 1,CSVファイルを確認(サンプルでDatabricksが用意しているデータ)
# MAGIC %fs ls dbfs:/databricks-datasets/learning-spark-v2/flights/departuredelays.csv

# COMMAND ----------

# MAGIC %fs head dbfs:/databricks-datasets/learning-spark-v2/flights/departuredelays.csv

# COMMAND ----------

# スキーマの設定
schema_conf = '''
  date string,
  delay int,
  distance int,
  origin string,
  destination string
'''

# CSVファイルをspark DataFrameに読み込む
df = (
  spark.read
  .format('csv')
  .schema(schema_conf)
  .option('Header', True)
  .load('dbfs:/databricks-datasets/learning-spark-v2/flights/departuredelays.csv')
)

# DataFrameを確認する
display(df)

# COMMAND ----------

# 上記同様にDataFrameをプロットしてみる
# (セルの最下部に出てくるグラフのアイコンをクリックする)
display(df)

# COMMAND ----------

# DBTITLE 1,Deltaフォーマットで保存する
(
  df.write
  .format('delta')
  .mode('overwrite')
  .save('dbfs:/FileStore/delta/flights/departuredelays')
)

# COMMAND ----------

# DBTITLE 1,Deltaテーブルを定義する
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS flight_dep
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/FileStore/delta/flights/departuredelays'

# COMMAND ----------

# DBTITLE 1,SQLでテーブルを確認する
# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM flight_dep

# COMMAND ----------

# DBTITLE 1,データを分析・探索する(可視化など)
# MAGIC %sql
# MAGIC -- 読み込んだレコード数
# MAGIC SELECT COUNT(*) FROM flight_dep

# COMMAND ----------

# MAGIC %sql
# MAGIC -- originが"ABE"のレコードのみをピックアップ
# MAGIC 
# MAGIC SELECT * FROM flight_dep
# MAGIC WHERE origin = "ABE"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- originが"ABE"のレコードのカウント数
# MAGIC 
# MAGIC SELECT COUNT(*) FROM flight_dep
# MAGIC WHERE origin = "ABE"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- originが"ABE"のレコードの平均のdelay時間
# MAGIC 
# MAGIC SELECT origin, avg(delay) FROM flight_dep
# MAGIC WHERE origin = "ABE"
# MAGIC GROUP BY origin

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 全てのoriginに対して、平均のdelay時間
# MAGIC 
# MAGIC SELECT origin, avg(delay) FROM flight_dep
# MAGIC GROUP BY origin

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 全てのoriginに対して、平均のdelay時間
# MAGIC -- 大きい順に並べる
# MAGIC 
# MAGIC SELECT origin, avg(delay) FROM flight_dep
# MAGIC GROUP BY origin
# MAGIC ORDER BY avg(delay) DESC

# COMMAND ----------

# MAGIC %md
# MAGIC # 環境のクリーンアップ

# COMMAND ----------

# MAGIC %sql
# MAGIC -- テーブルの削除
# MAGIC DROP TABLE flight_dep

# COMMAND ----------

# DBTITLE 1,Deltaファイルを削除
# MAGIC %fs rm -r dbfs:/FileStore/delta/flights/departuredelays
