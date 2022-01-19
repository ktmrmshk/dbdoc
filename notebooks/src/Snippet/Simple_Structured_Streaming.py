# Databricks notebook source
# MAGIC %md # Streaming!

# COMMAND ----------

# DBTITLE 1,環境設定 - ファイルパス・テーブル名の設定
# 以下にユニーク値を設定してください
username = "ユニークな名前で置き換えてください！"
# 例: 
username = "kitamura1112"

base_path = f'/tmp/{username}/streaming_lab'
db_name = f'db_{username}'

dbutils.fs.rm(base_path, True)
spark.sql(f''' DROP DATABASE IF EXISTS `{db_name}` CASCADE; ''')
spark.sql(f''' CREATE DATABASE `{db_name}`; ''')
spark.sql(f''' use {db_name}; ''')

print(f'''
  * base_path: {base_path} 
  * database: {db_name}
''')

# COMMAND ----------

# MAGIC %md ## Streamingのデータソース
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/data_sources.png" width=250>
# MAGIC 
# MAGIC チュートリアルで使用するデータ: `/databricks-datasets/definitive-guide/data/activity-data/*.json`

# COMMAND ----------

# DBTITLE 1,ファイル一覧
# MAGIC %fs ls /databricks-datasets/definitive-guide/data/activity-data/

# COMMAND ----------

# DBTITLE 1,JSONファイルの中身を確認
# MAGIC %fs head /databricks-datasets/definitive-guide/data/activity-data/part-00018-tid-730451297822678341-1dda7027-2071-4d73-a0e2-7fb6a91e1d1f-0-c000.json

# COMMAND ----------

# スキーマの定義(StreamingはinferSchemaしている時間がないので、ユーザーがスキーマを与える必要がある)
data_schema='''
Arrival_Time long,
Creation_Time long,
Device string,
Index long,
Model string,
User string,
gt string,
x double,
y double,
z double
'''

# COMMAND ----------

# MAGIC %md ## 1. データソースからStreamingを受信する (Rawテーブル)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/raw_table.png" width=500>

# COMMAND ----------

# ファイルパスなどの指定
raw_delta_path    = base_path + '/raw.delta'
raw_chkpoint_path = base_path + '/raw.checkpoint'
raw_table_name = 'table_raw'
print(f'''
  * raw_delta_path: {raw_delta_path}
  * raw_chkpoint_path: {raw_chkpoint_path}
  * raw_table_name: {raw_table_name}
''')

# 読み込み側
df_raw = (
  spark
  .readStream
  .format( 'json' )
  .schema( data_schema )
  .option( 'maxFilesPerTrigger', 1 )
  .load('/databricks-datasets/definitive-guide/data/activity-data/*.json')
)

# COMMAND ----------

# 書き込み側
ret_raw = (
  df_raw
  .writeStream
  .format('delta')
  .option('checkpointLocation', raw_chkpoint_path)
  .option('path', raw_delta_path)
  .trigger(processingTime='3 seconds')
  .outputMode('append')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {raw_table_name}
  USING delta
  LOCATION '{raw_delta_path}';
''')

# COMMAND ----------

display(
  spark.sql(f'''
    SELECT * FROM {raw_table_name}
  ''')
)

# COMMAND ----------

# MAGIC %md ## 2. データのクレンジング(シルバーテーブル)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/silver_table.png" width=700>

# COMMAND ----------

# DBTITLE 1,User属性テーブルを事前に用意(Dimentionテーブル、静的)
import json

user_dimension=json.loads('''
[
{ "user_id": "a", "state": "東京"},
{ "user_id": "b", "state": "千葉"},
{ "user_id": "c", "state": "神奈川"},
{ "user_id": "d", "state": "埼玉"},
{ "user_id": "e", "state": "栃木"},
{ "user_id": "f", "state": "群馬"},
{ "user_id": "g", "state": "長野"},
{ "user_id": "h", "state": "茨城"},
{ "user_id": "i", "state": "山梨"}
]
''')

df_user = spark.createDataFrame(user_dimension)
display(df_user)

# COMMAND ----------

# ファイルパスなどの指定
cleaned_delta_path    = base_path + '/cleaned.delta'
cleaned_chkpoint_path = base_path + '/cleaned.checkpoint'
cleaned_table_name = 'table_cleaned'
print(f'''
  * cleaned_delta_path: {cleaned_delta_path}
  * cleaned_chkpoint_path: {cleaned_chkpoint_path}
  * cleaned_table_name: {cleaned_table_name}
''')


# 読み込み側
df_raw = (
  spark
  .readStream
  .format( 'delta' )
  .load(raw_delta_path) # rawテーブル(delta)のパスから読み込む
)

# COMMAND ----------

# クレンジング処理 + Join処理
df_cleaned = (
  df_raw
  .withColumn('Arrival_ts', to_timestamp( col('Arrival_Time') / 1000.0 )  )
  .withColumn('Creation_ts', to_timestamp( col('Creation_Time') / 1000000000.0 )  )

  .select('Arrival_ts', 'Creation_ts', 'Device', 'Index', 'Model', 'User', 'gt', 'x', 'y', 'z')
  .join(df_user, df_raw.User == df_user.user_id, "left_outer" )
)

# 書き込み
ret_cleaned = (
  df_cleaned
  .writeStream
  .format('delta')
  .option('checkpointLocation', cleaned_chkpoint_path)
  .option('path', cleaned_delta_path)
  .outputMode('append')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {cleaned_table_name}
  USING delta
  LOCATION '{cleaned_delta_path}';
''')

# COMMAND ----------

# DBTITLE 1,テーブルの確認(SQL編)
# MAGIC %sql 
# MAGIC -- テーブルの確認(SQL編)
# MAGIC 
# MAGIC SELECT * FROM table_cleaned

# COMMAND ----------

# DBTITLE 1,テーブルの確認(Python編)
df_tmp = spark.read.format('delta').load(cleaned_delta_path)

display( df_tmp )

# COMMAND ----------

# MAGIC %md ## 3. ビジネスサマリテーブル
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/gold_table.png" width=1000>

# COMMAND ----------

# ファイルパスなどの指定
summary_delta_path    = base_path + '/summary.delta'
summary_chkpoint_path = base_path + '/summary.checkpoint'
summary_table_name = 'table_summary'
print(f'''
  * summary_delta_path: {summary_delta_path}
  * summary_chkpoint_path: {summary_chkpoint_path}
  * summary_table_name: {summary_table_name}
''')


# 読み込み側
df_cleaned = (
  spark
  .readStream
  .format( 'delta' )
  .load(cleaned_delta_path) # cleanedテーブル(delta)のパスから読み込む
)

# COMMAND ----------

# 集約・統計
df_summary = (
  df_cleaned
  .groupBy('User', 'Device', 'state')
  .agg( 
    count('*').alias('count'),
    sum('x').alias('sum_x'), 
    mean('y').alias('mean_y'),
    stddev('z').alias('stddev_z'),
    max('Index').alias('max_index'), 
    percentile_approx('z', 0.5).alias('median_z')
  )
)



# COMMAND ----------

# 書き出し
ret_summary = (
  df_summary
  .writeStream
  .format('delta')
  .option('checkpointLocation', summary_chkpoint_path)
  .option('path', summary_delta_path)
  .outputMode('complete')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {summary_table_name}
  USING delta
  LOCATION '{summary_delta_path}';
''')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM table_summary;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY table_summary;

# COMMAND ----------

# MAGIC %md ## 環境のクリーンアップ

# COMMAND ----------

dbutils.fs.rm(base_path, True)
spark.sql(f''' DROP DATABASE IF EXISTS `{db_name}` CASCADE; ''')

# COMMAND ----------


