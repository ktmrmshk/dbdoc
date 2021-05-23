# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## 1. 各種関数を定義する
# MAGIC 
# MAGIC 本来は別notebookにすべきだが、簡単のため同じnotebookの前半で実施する

# COMMAND ----------

# DBTITLE 1,1. Autoloaderで作成したSQSのキューを削除する関数
# MAGIC %scala
# MAGIC import com.databricks.sql.CloudFilesAWSResourceManager
# MAGIC 
# MAGIC def deleteNotifications(): Unit = {
# MAGIC   val manager = CloudFilesAWSResourceManager
# MAGIC     .newManager
# MAGIC     .option("cloudFiles.region", "ap-northeast-1")
# MAGIC     .create()
# MAGIC 
# MAGIC   val ret = manager.listNotificationServices()
# MAGIC   
# MAGIC   val streamList = ret.select("streamId").rdd.map(r => r(0)).collect.toList
# MAGIC   for(st <- streamList){
# MAGIC     println(st)
# MAGIC     manager.tearDownNotificationServices(st.asInstanceOf[String])
# MAGIC   }
# MAGIC }

# COMMAND ----------

# DBTITLE 1,2. Deltaテーブル、ファイルの削除
# テーブルおよびファイルを削除する処理(何回も試す場合に使用)
def cleanup(db_name, table_name, delta_path, checkpoint_path):
  db_name=db_name.replace('-','_')
  table_name=table_name.replace('-','_')
  # tableの削除
  try:
    spark.sql(f'DROP TABLE `{db_name}`.`{table_name}`')
  except Exception as e:
    print(e)
  
  # fileの削除
  try:
    dbutils.fs.rm(delta_path, recurse=True)
  except Exception as e:
    print(e)
 
  try:
    dbutils.fs.rm(checkpoint_path, recurse=True)
  except Exception as e:
    print(e)
    

# COMMAND ----------

# DBTITLE 1,3. Autoloaderの作成
def create_autoloader(schema_conf, csv_path, delta_path, checkpoint_path, db_name, table_name):
  # autoloaderでS3バケツからcsvファイルをロードする
  df = (
    spark.readStream
    .format('cloudFiles')
    .option('cloudFiles.format', 'csv')
    .option('cloudFiles.useNotifications', True)
    .option('cloudFiles.region', 'ap-northeast-1')
    .schema(schema_conf)
    .load(csv_path)
  )

  # loadしたデータをdeltaに書き込む
  (
    df.writeStream
    .format('delta')
    .option('checkpointLocation', checkpoint_path)
    .start(delta_path)
  )
  
def create_delta_table(delta_path, db_name, table_name):
  
  db_name=db_name.replace('-','_')
  table_name=table_name.replace('-','_')
  
  # Delta Tableに登録する
  spark.sql(f'''
    CREATE DATABASE IF NOT EXISTS `{db_name}`;
  ''')
  
  spark.sql(f'''
    CREATE TABLE IF NOT EXISTS `{db_name}`.`{table_name}`
    USING delta
    LOCATION "{delta_path}";
  ''')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2. 各種センサーデータに適用する

# COMMAND ----------

# MAGIC %md
# MAGIC ### 補足(DB名とTable名)
# MAGIC DB名、テーブル名は以下の通りです。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC #### ストリーミング処理
# MAGIC 
# MAGIC | DB名 | Table名 | Schema |
# MAGIC |------|---------|--------|
# MAGIC | Machine-A | sensor001 | `datetime timestamp`, `sensor_data int` |
# MAGIC | Machine-A | sensor002 | `datetime timestamp`, `sensor_data int` |
# MAGIC | Machine-A | sensor003 | `datetime timestamp`, `sensor_data int` |
# MAGIC | Machine-B | sensor001 | `datetime timestamp`, `sensor_data int` |
# MAGIC | Machine-B | sensor002 | `datetime timestamp`, `sensor_data int` |
# MAGIC | Machine-C | sensor001 | `datetime timestamp`, `sensor_data int` |
# MAGIC | ... | ... | ... |

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Machine-A

# COMMAND ----------

sensor_type = 'Machine-A'
sensor_pos = ['001', '002']

csv_path_base = 's3a://abc123-corp-iot-raw/'
delta_path_base = 's3a://abc123-corp-iot-raw/databricks/project123/delta'
checkpoint_path_base = 's3a://abc123-corp-iot-raw/databricks/project123/checkpoint'

for sp in sensor_pos:
  
  schema_conf='''
  datetime timestamp,
  sensor_data int
  '''
  
  csv_path = f'{csv_path_base}/{sensor_type}/{sp}/202104*.csv'
  delta_path = f'{delta_path_base}/{sensor_type}/{sp}'
  checkpoint_path = f'{checkpoint_path_base}/{sensor_type}/{sp}'
  
  db_name=f'{sensor_type}'
  table_name=f'{sp}'
  
  print(f'csv_path => {csv_path}')
  print(f'delta_path => {delta_path}')
  print(f'checkpoint_path => {checkpoint_path}')
  print(f'db_name => {db_name}')
  print(f'table_name => {table_name}')
  
  create_autoloader(schema_conf, csv_path, delta_path, checkpoint_path, db_name, table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### プロット例
# MAGIC 
# MAGIC 以下のSQLで`Machine-A`の`001, 002`のセンサデータを一つのテーブルにまとめて一つのグラフでプロットする。
# MAGIC 
# MAGIC **時系列をそのままプロット**
# MAGIC ```
# MAGIC -- 時系列をそのままプロット
# MAGIC select
# MAGIC   *,
# MAGIC   "001" as sensor_type,
# MAGIC   date_trunc('HOUR', datetime) as ts_hour,
# MAGIC   date_trunc('DAY', datetime) as ts_day
# MAGIC from
# MAGIC   machineA.sensor001
# MAGIC union
# MAGIC select
# MAGIC   *,
# MAGIC   "002" as sensor_type,
# MAGIC   date_trunc('HOUR', datetime) as ts_hour,
# MAGIC   date_trunc('DAY', datetime) as ts_day
# MAGIC from
# MAGIC   machineA.sensor002
# MAGIC order by datetime
# MAGIC ```
# MAGIC 
# MAGIC **1時間毎のデータ数**
# MAGIC ```
# MAGIC -- 1時間毎のデータ数
# MAGIC select ts_hour, sensor_type, count(sensor_data)
# MAGIC from
# MAGIC (
# MAGIC   select
# MAGIC     *,
# MAGIC     "001" as sensor_type,
# MAGIC     date_trunc('HOUR', datetime) as ts_hour,
# MAGIC     date_trunc('DAY', datetime) as ts_day
# MAGIC   from
# MAGIC     machineA.sensor001
# MAGIC   union
# MAGIC   select
# MAGIC     *,
# MAGIC     "002" as sensor_type,
# MAGIC     date_trunc('HOUR', datetime) as ts_hour,
# MAGIC     date_trunc('DAY', datetime) as ts_day
# MAGIC   from
# MAGIC     machineA.sensor002
# MAGIC )
# MAGIC group by ts_hour, sensor_type
# MAGIC order by ts_hour, sensor_type
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Machine-B

# COMMAND ----------

# ここにコードを書いてください

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Machine-C

# COMMAND ----------

# ここにコードを書いてください

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Machine-D

# COMMAND ----------

# ここにコードを書いてください

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. (必要に応じて)環境のクリーンアップ(リセット)

# COMMAND ----------

# sensor_type = 'machineA'
# sensor_pos = ['001', '002']

# #csv_path_base = 's3a://abc123-corp-iot-raw/'
# delta_path_base = 's3a://abc123-corp-iot-raw/databricks/project123/delta'
# checkpoint_path_base = 's3a://abc123-corp-iot-raw/databricks/project123/checkpoint'

# for sp in sensor_pos:
  
#   #csv_path = f'{csv_path_base}/{sensor_type}/{sp}/2021*.csv'
#   delta_path = f'{delta_path_base}/{sensor_type}/{sp}'
#   checkpoint_path = f'{checkpoint_path_base}/{sensor_type}/{sp}'
  
#   db_name=f'{sensor_type}'
#   table_name=f'{sp}'
  
#   # clean-up
#   cleanup(db_name, table_name, delta_path, checkpoint_path)


# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC // aws sqsの削除
# MAGIC //deleteNotifications()
