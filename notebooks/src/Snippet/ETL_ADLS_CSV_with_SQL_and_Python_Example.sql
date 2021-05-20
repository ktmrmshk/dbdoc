-- Databricks notebook source
-- MAGIC %md
-- MAGIC 
-- MAGIC # データソースへのアクセス
-- MAGIC 
-- MAGIC Databricks上でAzure Data Lake Storage(ADLS)と連携する場合、以下の3通りの方法があります。
-- MAGIC 
-- MAGIC 1. ADLSのアクセスキーを用いてアクセスする
-- MAGIC 1. ADDのサービスプリンシパルを用いてアクセスする
-- MAGIC 1. SAS(トークン)を用いてアクセスする
-- MAGIC 
-- MAGIC ここでは、ADLSのアクセスキーを用いてアクセスする方法をみていきます。アクセスキーがnotebook上に平文で書かれるため、取り扱いに注意してください。
-- MAGIC (本番環境ではSecret機能を使ってアクセスキーを管理することが推奨されます)

-- COMMAND ----------

-- DBTITLE 0,ストレージへ直接アクセス
-- MAGIC %python
-- MAGIC 
-- MAGIC # ストレージアカウント、コンテナ、アクセスキーの設定
-- MAGIC storage_account = 'your_ADLS_storage_account'
-- MAGIC storage_container = 'your_ADLS_storage_container'
-- MAGIC access_key = 'Your-ADLS-Container-Access-Key'
-- MAGIC 
-- MAGIC # spark環境変数に上記の内容を反映
-- MAGIC spark.conf.set( f'fs.azure.account.key.{storage_account}.dfs.core.windows.net', access_key)
-- MAGIC 
-- MAGIC # File Path Base
-- MAGIC print( f'File Path Base=>"abfss://{storage_container}@{storage_account}.dfs.core.windows.net/" \n')
-- MAGIC 
-- MAGIC # ストレージにアクセスできるか確認 (list)
-- MAGIC dbutils.fs.ls(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/')

-- COMMAND ----------

-- DBTITLE 1,DBFSコマンドでファイルを一覧を取得する
-- MAGIC %fs ls abfss://<storage_container>@<storage_account>.dfs.core.windows.net/

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # CSVをSQLでロード・Deltaテーブル化する

-- COMMAND ----------

-- DBTITLE 1,Databricksが用意しているサンプルのCSVファイルを使用する
-- MAGIC %fs ls /databricks-datasets/flights/

-- COMMAND ----------

-- MAGIC %fs head /databricks-datasets/flights/departuredelays.csv

-- COMMAND ----------

-- DBTITLE 1,CSVをtemp viewで読み込む
CREATE
OR REPLACE TEMP VIEW temp123 (
  flihgt_date String,
  delay Int,
  distance Int,
  origin String,
  destination String
) USING CSV OPTIONS (
  path "/databricks-datasets/flights/departuredelays.csv",
  header "true"
)

-- COMMAND ----------

--- Viewの内容の確認
SELECT * FROM temp123

-- COMMAND ----------

--- Schemaの確認
DESCRIBE temp123

-- COMMAND ----------

--- カーディナリティの確認
SELECT COUNT( DISTINCT origin) FROM temp123

-- COMMAND ----------

-- DBTITLE 1,ETLを実施して、かつ、結果をDELTA Tableに書き出す
-- 注意: SQLでは変数が使用できないため、<storage_container>および<storage_account>を適宜置換した後、実行してください。

DROP TABLE IF EXISTS flight_record;

CREATE TABLE flight_record
USING DELTA
PARTITIONED BY (origin)
LOCATION "abfss://<storage_container>@<storage_account>.dfs.core.windows.net/databricks-sandbox/flight_record"
AS (
  SELECT 
    to_timestamp(flihgt_date, 'MMddHHmm') AS flight_timestamp,
    delay,
    distance,
    origin,
    destination
  FROM temp123
)



-- COMMAND ----------

-- DBTITLE 1,ストレージに書き出されたDeltaファイルを確認する
-- MAGIC %fs ls abfss://<storage_container>@<storage_account>.dfs.core.windows.net/databricks-sandbox/flight_record

-- COMMAND ----------

--- SchemaとPartitionの確認
DESCRIBE flight_record

-- COMMAND ----------

-- DBTITLE 1,簡単な可視化
SELECT origin, destination, AVG(delay), AVG(distance)
FROM flight_record
GROUP BY origin, destination
ORDER BY AVG(delay) desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC # CSVをPythonでロードしてDeltaテーブル化する

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC schema_string = '''
-- MAGIC   flihgt_date String,
-- MAGIC   delay Int,
-- MAGIC   distance Int,
-- MAGIC   origin String,
-- MAGIC   destination String
-- MAGIC '''
-- MAGIC 
-- MAGIC DataFrame_Bronz = (
-- MAGIC   spark.read
-- MAGIC   .format('csv')
-- MAGIC   .option('Header', True)
-- MAGIC   .schema(schema_string)
-- MAGIC   .load('/databricks-datasets/flights/departuredelays.csv')
-- MAGIC )
-- MAGIC 
-- MAGIC display(DataFrame_Bronz)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import *
-- MAGIC display( DataFrame_Bronz.withColumn('foobar',  col('distance') - col('delay') ) )

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import *
-- MAGIC 
-- MAGIC DataFrame_Silver = (
-- MAGIC   DataFrame_Bronz
-- MAGIC   .select('flihgt_date', 'delay', 'distance', 'origin', 'destination')
-- MAGIC   .withColumn('flight_timestamp', to_timestamp('flihgt_date', 'MMddHHmm'))
-- MAGIC   .drop('flihgt_date')
-- MAGIC )
-- MAGIC 
-- MAGIC display(DataFrame_Silver)
-- MAGIC 
-- MAGIC # Deltaで保存する
-- MAGIC (
-- MAGIC   DataFrame_Silver
-- MAGIC   .write
-- MAGIC   .partitionBy('origin')
-- MAGIC   .format('delta')
-- MAGIC   .mode('overwrite')
-- MAGIC   .save(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/databricks-sandbox/flight_record_py')
-- MAGIC )

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC 
-- MAGIC spark.sql(f'''
-- MAGIC   CREATE TABLE IF NOT EXISTS flight_record_py
-- MAGIC   USING DELTA
-- MAGIC   LOCATION "abfss://{storage_container}@{storage_account}.dfs.core.windows.net/databricks-sandbox/flight_record_py"
-- MAGIC ''')
-- MAGIC 
-- MAGIC df = spark.sql('''
-- MAGIC   SELECT * FROM flight_record_py
-- MAGIC ''')
-- MAGIC 
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC df = spark.sql('''
-- MAGIC   DESCRIBE history flight_record_py
-- MAGIC ''')
-- MAGIC 
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC # (オプション) Deltaテーブルのクリア・削除

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # ファイルを削除
-- MAGIC dbutils.fs.rm(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/databricks-sandbox/flight_record_py', True)
-- MAGIC 
-- MAGIC # テーブルをDROP
-- MAGIC spark.sql('''
-- MAGIC   DROP TABLE flight_record_py
-- MAGIC ''')
