# Databricks notebook source
# データの読み込み

df_bronze = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .option('inferSchema', True) # schemaは推定
  #.load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-*.csv.gz')
  .load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-10.csv.gz')
)

# 読み込んだデータフレームを確認
display( df_bronze )

# COMMAND ----------

from pyspark.sql.functions import col, to_date, year, month, dayofweek, minute, second, hour, date_format

# timestampのcolumnからパーティションのためのcolumnを作る
# サンプルのため、いろいろなパターンの日時変換の例をつけてあります。
df_silver = (
  df_bronze
  .withColumn('to_date'  , to_date(   col('tpep_pickup_datetime') ))
  .withColumn('year'     , year(      col('tpep_pickup_datetime') ))
  .withColumn('month'    , month(     col('tpep_pickup_datetime') ))
  .withColumn('dayofweek', dayofweek( col('tpep_pickup_datetime') ))
  .withColumn('hour'     , hour(      col('tpep_pickup_datetime') ))
  .withColumn('date_format', date_format('tpep_pickup_datetime', 'yyyy, MM-dd_HH:mm:ss')  )
)

display( df_silver )

# COMMAND ----------

# 日時ごとにPartitionする場合(パーティション後のファイルサイズが1GB以上出ないとパフォーマンスが劣化する可能性があります)

(
  df_silver
  .write
  .format('delta')
  .partitionBy('to_date') # partitionで用いるColを指定する
  .save('/tmp/partition_example/lending_club/silver.delta')
)

# COMMAND ----------

# ファイルのlist (フォルダがparitionで切られている)
display(
  dbutils.fs.ls('/tmp/partition_example/lending_club/silver.delta') 
)

# COMMAND ----------

# 1つファイルサイズを確認
# => 10 KB
# => 今回の例では細かくパーティションに分け過ぎなので、パフォーマンスは良くない

display(
  dbutils.fs.ls('/tmp/partition_example/lending_club/silver.delta/to_date=2019-09-30/') 
)

# COMMAND ----------

#ファイルのクリア(環境のクリーンアップ)
dbutils.fs.rm('/tmp/partition_example/lending_club/silver.delta/', True)
