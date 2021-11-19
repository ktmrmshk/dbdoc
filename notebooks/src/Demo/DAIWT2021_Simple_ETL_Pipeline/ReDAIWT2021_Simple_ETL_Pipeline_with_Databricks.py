# Databricks notebook source
# 初期化設定
sql('set spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;')
sql('set spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;')
#dbutils.fs.rm('/tmp/daiwt2021/', True)

# COMMAND ----------

# MAGIC %md ##CSVの読み込み
# MAGIC 
# MAGIC CSV File => `/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv`
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/csv_to_delta1.png" width=400  >

# COMMAND ----------

df = (
  spark.read.format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .load('/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv') 
)

df.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/loan_stats.delta')

# COMMAND ----------

df_delta = spark.read.format('delta').load('/tmp/daiwt2021/loan_stats.delta')
display(df_delta)

# COMMAND ----------

dbutils.data.summarize( df_delta )

# COMMAND ----------

# MAGIC %md ## ETL・データ整形
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/csv_to_delta2.png" width=600>

# COMMAND ----------

# 簡単な処理を行い、結果を保存
from pyspark.sql.functions import col, expr

df_raw = spark.read.format('delta').load('/tmp/daiwt2021/loan_stats.delta')

(
  df_raw
  .select('loan_amnt', # 必要なカラムの抽出
            'term',
            'int_rate',
            'grade',
            'addr_state',
            'emp_title',
            'home_ownership',
            'annual_inc',
            'loan_status')
  .withColumn('int_rate', expr('cast(replace(int_rate,"%","") as float)')) # データ型の変換
  .withColumnRenamed('addr_state', 'state') # カラム名変更
  .write
  .format('delta') #　Deltaで保存(silverテーブル)
  .mode('overwrite')
  .save('/tmp/daiwt2021/loan_stat_silver.delta')
)

display( spark.read.format('delta').load('/tmp/daiwt2021/loan_stat_silver.delta') )

# COMMAND ----------

# MAGIC %md ## SQLも使えます・可視化もそのままできます

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS daiwt2021_kitamura CASCADE;
# MAGIC CREATE DATABASE daiwt2021_kitamura;
# MAGIC USE daiwt2021_kitamura;
# MAGIC 
# MAGIC CREATE TABLE loan_stat_silver
# MAGIC USING delta
# MAGIC LOCATION '/tmp/daiwt2021/loan_stat_silver.delta';
# MAGIC 
# MAGIC SELECT state, loan_status, count(*) as counts 
# MAGIC FROM loan_stat_silver
# MAGIC GROUP BY state, loan_status
# MAGIC ORDER BY counts DESC

# COMMAND ----------

# MAGIC %md ## SQLの結果をPythonで受け取り、サマリテーブルを作成する
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/csv_to_delta3.png" width=700>

# COMMAND ----------

df_gold = spark.sql('''

SELECT state, loan_status, count(*) as counts 
FROM loan_stat_silver
GROUP BY state, loan_status
ORDER BY counts DESC

''')

# Deltaで保存(goldテーブル)
df_gold.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/loan_stat_gold.delta')

# goldテーブルをHiveへ登録
spark.sql('''
  CREATE TABLE IF NOT EXISTS loan_stat_gold
  USING delta
  LOCATION '/tmp/daiwt2021/loan_stat_gold.delta'
''')

# COMMAND ----------

# MAGIC %md ## DeltaテーブルをBIから参照する
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/delta_to_bi.png" width="900">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC BIツールから接続するDeltaテーブルのJDBC/ODBCのエンドポイント
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/bi_endpoint.png" width=1000>

# COMMAND ----------

# MAGIC %md ## Deltaテーブルの内容をDWH/RDBMSに書き出す
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/delta_to_rdb.png" width="900">

# COMMAND ----------

mysql_url = f'jdbc:mysql://{hostname}:{port}/{database}?user={username}&password={password}'

df_gold = spark.read.format('delta').load('/tmp/daiwt2021/loan_stat_gold.delta')

(
  df_gold.write.format("jdbc")
  .option("url", mysql_url)
  .option("dbtable", "test")
  .mode("overwrite")
  .save()
)

# COMMAND ----------

# MAGIC %md ## Deltaテーブルの内容をファイルに書き出す
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/delta_to_files.png" width="900">

# COMMAND ----------

df_gold = spark.read.format('delta').load('/tmp/daiwt2021/loan_stat_gold.delta')

df_gold.write.format('csv'     ).save('/tmp/daiwt2021/loan_stat_gold.csv')
df_gold.write.format('json'    ).save('/tmp/daiwt2021/loan_stat_gold.json')
df_gold.write.format('parquet' ).save('/tmp/daiwt2021/loan_stat_gold.parquet')
df_gold.write.format('avro'    ).save('/tmp/daiwt2021/loan_stat_gold.avro')

# COMMAND ----------

# MAGIC %md ## 今までのETLのコードをまとめる
# MAGIC 
# MAGIC 以下のコードでシンプルにETL処理が可能です。
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/whole_etl.png" width=700>

# COMMAND ----------

### 1. Rawテーブル (CSV => bornze)
df_raw = (
  spark.read.format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .load('/databricks-datasets/lending-club-loan-stats/LoanStats_2018Q2.csv') 
)
df_raw.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/loan_stats.delta')
sql("CREATE TABLE loan_stat_raw USING delta LOCATION '/tmp/daiwt2021/loan_stat.delta';")


### 2. データ整形 (bronze => silver)
from pyspark.sql.functions import col, expr

df_silver = (
  spark.read.format('delta').load('/tmp/daiwt2021/loan_stats.delta')
  .select('loan_amnt', 'term', 'int_rate', 'grade', 'addr_state', 'emp_title', 'home_ownership', 'annual_inc', 'loan_status')
  .withColumn('int_rate', expr('cast(replace(int_rate,"%","") as float)'))
  .withColumnRenamed('addr_state', 'state')
)
df_silver.write.format('delta').mode('overwrite').save( '/tmp/daiwt2021/loan_stat_silver.delta')
sql("CREATE TABLE loan_stat_silver USING delta LOCATION '/tmp/daiwt2021/loan_stat_silver.delta';"")

    
### 3. サマリテーブル (silver => gold)
df_gold = spark.sql('''
  SELECT state, loan_status, count(*) as counts 
  FROM loan_stat_silver
  GROUP BY state, loan_status
  ORDER BY counts DESC
''')

df_gold.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/loan_stat_gold.delta')
sql("CREATE TABLE IF NOT EXISTS loan_stat_gold USING delta LOCATION '/tmp/daiwt2021/loan_stat_gold.delta'")

# COMMAND ----------

# MAGIC %md ## JSONを読み込む
# MAGIC 
# MAGIC JSON File => `/mnt/training/ecommerce/events/events-500k.json`
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/json_to_delta.png" width=400>

# COMMAND ----------

df_json = (
  spark.read.format('json')
  .option('inferSchema', True)
  .load('/mnt/training/ecommerce/events/events-500k.json')
)

# deltaへの書き込み
df_json.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/events.delta')

# 確認
display( spark.read.format('delta').load('/tmp/daiwt2021/events.delta') )

# COMMAND ----------

# MAGIC %md ## LOGデータを読み込む
# MAGIC 
# MAGIC 任意のテキストフォーマットでも対応可能です。
# MAGIC 
# MAGIC access.log(Webサーバーアクセスログ) => `s3://databricks-ktmr-s3/var/log/access.log.*.gz`
# MAGIC 
# MAGIC ----
# MAGIC ```
# MAGIC 13.66.139.0 - - [19/Dec/2020:13:57:26 +0100] "GET /index.php?option=com_phocagallery&view=category&id=1:almhuette-raith&Itemid=53 HTTP/1.1" 200 32653 "-" "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)" "-"
# MAGIC 157.48.153.185 - - [19/Dec/2020:14:08:06 +0100] "GET /apache-log/access.log HTTP/1.1" 200 233 "-" "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36" "-"
# MAGIC 157.48.153.185 - - [19/Dec/2020:14:08:08 +0100] "GET /favicon.ico HTTP/1.1" 404 217 "http://www.almhuette-raith.at/apache-log/access.log" "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36" "-"
# MAGIC ...
# MAGIC ```
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/log_to_delta.png" width=400>

# COMMAND ----------

from pyspark.sql.functions import split, regexp_extract, col, to_timestamp

raw_df = spark.read.text('s3://databricks-ktmr-s3/var/log/access.log.*.gz')

# Regexでログデータをパース
split_df = (
    raw_df.select(
      regexp_extract('value', r'^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', 1).alias('src_ip'),
      regexp_extract('value', r'\[(.+?)\]', 1).alias('time_string'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 1).alias('method'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 2).alias('path'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 3).alias('version'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+)', 1).cast('int').alias('status_code'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+)', 2).cast('int').alias('content_size'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+) "(.+?)" "(.+?)" "(.+?)"', 3).alias('host2'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+) "(.+?)" "(.+?)" "(.+?)"', 4).alias('user_agent')
    )
  .withColumn( 'timestamp', to_timestamp(  col('time_string'), 'dd/MMM/yyyy:HH:mm:ss Z') )
  .drop('time_string')
  .filter( col('timestamp').isNotNull() ) 
)

# Deltaに書き出す
split_df.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/access_log.delta')

# 確認
display( spark.read.format('delta').load('/tmp/daiwt2021/access_log.delta') )

# COMMAND ----------

# MAGIC %md ## 画像ファイル(バイナリファイル)を読み込む
# MAGIC 
# MAGIC Image Files => `/databricks-datasets/cctvVideos/train_images/label=0/*.jpg`
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/binary_to_delta.png" width=400>

# COMMAND ----------

image_df = (
  spark.read.format('binaryFile')
  .option('mimeType', 'image/*')
  .load('/databricks-datasets/cctvVideos/train_images/label=0/*.jpg')
)

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
image_df.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/train_image.delta')
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

# 確認
display( spark.read.format('delta').load('/tmp/daiwt2021/train_image.delta') ) 

# COMMAND ----------

# MAGIC %md ## RDBMS/DWHから読み込む
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/rdb_to_delta.png" width=400>

# COMMAND ----------

jdbcHostname = "server1.databricks.training"
jdbcPort = 5432
jdbcDatabase = "training"
jdbcUrl = f"jdbc:postgresql://{jdbcHostname}:{jdbcPort}/{jdbcDatabase}"


query = '''
(
  SELECT * FROM training.people_1m  
  WHERE salary > 100000
) emp_alias
'''

df = spark.read.jdbc(url=jdbcUrl, table=query, properties=connectionProps)
df.write.format('delta').mode('overwrite').save('/tmp/daiwt2021/salary.delta')

# 確認
display( spark.read.format('delta').load('/tmp/daiwt2021/salary.delta') )

# COMMAND ----------

# MAGIC %md ## ストリーミングを読み込む
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/streaming_to_delta.png" width=400>

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col
import re

#　JSONデータのスキーマ定義
schema = StructType([
  StructField("channel", StringType(), True),
  StructField("comment", StringType(), True),
  StructField("delta", IntegerType(), True),
  StructField("flag", StringType(), True),
  StructField("geocoding", StructType([ StructField("city", StringType(), True), StructField("country", StringType(), True), StructField("countryCode2", StringType(), True), StructField("countryCode3", StringType(), True), StructField("stateProvince", StringType(), True), StructField("latitude", DoubleType(), True), StructField("longitude", DoubleType(), True), ]), True),
  StructField("isAnonymous", BooleanType(), True),
  StructField("isNewPage", BooleanType(), True),
  StructField("isRobot", BooleanType(), True),
  StructField("isUnpatrolled", BooleanType(), True),
  StructField("namespace", StringType(), True),         
  StructField("page", StringType(), True),              
  StructField("pageURL", StringType(), True),           
  StructField("timestamp", StringType(), True),        
  StructField("url", StringType(), True),
  StructField("user", StringType(), True),              
  StructField("userURL", StringType(), True),
  StructField("wikipediaURL", StringType(), True),
  StructField("wikipedia", StringType(), True),
])

#　読み込み
stream_df = (
  spark.readStream.format('kafka') # Kafkaをソースと指定
  .option('kafka.bootstrap.servers', 'server2.databricks.training:9092')
  .option('subscribe', 'en')
  .load()
)

# ELTをして、Deltaに書き込む
(
  stream_df
  .withColumn('json', from_json(col('value').cast('string'), schema))   # Kafkaのバイナリデータを文字列に変換し、from_json()でJSONをパース
  .select(col("json.*"))                    # JSONの子要素だけを取り出す
  .writeStream                              # writeStream()でストリームを書き出す
  .format('delta')                          # Deltaとして保存
  .option('checkpointLocation', '/tmp/daiwt2021/stream.checkpoint') # チェックポイント保存先を指定
  .outputMode('append')                     # マイクロバッチの結果をAppendで追加
  .start('/tmp/daiwt2021/stream.delta')   # start()でストリーム処理を開始 (アクション)
)

# COMMAND ----------

# 確認
df = spark.readStream.format('delta').load('/tmp/daiwt2021/stream.delta')
display( df )

# COMMAND ----------

# MAGIC %md ## 随時更新追加されるファイルをストリーミングとして扱う
# MAGIC 
# MAGIC ファイルのディレクトリ => `s3a://databricks-ktmr-s3/stocknet-dataset/tweet/raw/AAPL/*`
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * オブジェクトストレージ上にファイルが随時アップロードされるパターンもよくあります。
# MAGIC * Databricksでは、ストレージに新たな追加されたファイルを認識して、そのファイルのみ読み込むことができます。
# MAGIC * この場合、ストレージをストリーミングのソースとして利用することになります。
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/images/autoloader.png" width=400>

# COMMAND ----------

df=spark.read.format('json').load('s3a://databricks-ktmr-s3/stocknet-dataset/tweet/raw/AAPL/2014-01-01')
tweet_schema = df.schema

# COMMAND ----------

df_autoloader = (
  spark.readStream.format('cloudFiles')
  .option('cloudFiles.format', 'json')
  .option('cloudFiles.maxBytesPerTrigger', '50KB') # 一度に読むサイズ上限
  .schema(tweet_schema)
  .load('s3a://databricks-ktmr-s3/stocknet-dataset/tweet/raw/AAPL/*')
)

(
  df_autoloader.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/daiwt2021/tweet.checkpoint')
  .option('maxFilesPerTrigger', 25) # 一度に読むファイル数上限
  .outputMode('append')
  .trigger(processingTime='2 seconds') # 2秒に一度処理
  #.trigger(once=True) # 一度だけ処理
  .start('/tmp/daiwt2021/tweet.delta')
  #.awaitTermination() # async => sync
)

# COMMAND ----------

# 確認
spark.read.format('delta').load('/tmp/daiwt2021/tweet.delta').count()

# COMMAND ----------

# MAGIC %md ##Deltaのパフォーマンス

# COMMAND ----------

taxi_schema='''
  vendor_id string,
  pickup_datetime timestamp,
  dropoff_datetime timestamp,
  passenger_count int,
  trip_distance double,
  pickup_longitude double,
  pickup_latitude double,
  rate_code string,
  store_and_fwd_flag string,
  dropoff_longitude double,
  dropoff_latitude double,
  payment_type string,
  fare_amount double,
  surcharge double,
  mta_tax string,
  tip_amount double,
  tolls_amount double,
  total_amount double
'''

df = (
  spark.read.format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .schema(taxi_schema)
  .load('/databricks-datasets/nyctaxi/tripdata/yellow/*.csv.gz')
)

# COMMAND ----------

df.write.format('delta').save('/tmp/daiwt2021_lts/yellow_taxi.delta')

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE delta.`/tmp/daiwt2021_lts/yellow_taxi.delta`　ZORDER BY (vendor_id, passenger_count, payment_type)

# COMMAND ----------

df_delta = spark.read.format('delta').load('/tmp/daiwt2021_lts/yellow_taxi.delta')
print( 'Record Count => {:,}'.format(df_delta.count() ) )

# COMMAND ----------

df = spark.read.format('delta').load('/tmp/daiwt2021_lts/yellow_taxi.delta')
display(df)

# COMMAND ----------

# フルスキャンの集約クエリ
display(
  df.groupBy('vendor_id','passenger_count','payment_type')
  .count()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE delta.`/tmp/daiwt2021_lts/yellow_taxi.delta` ZORDER BY (vendor_id, passenger_count, payment_type)

# COMMAND ----------

display(
  df.groupBy('vendor_id','passenger_count','payment_type')
  .count()
)

# COMMAND ----------

# MAGIC %md ## Job・自動実行・パイプライン化
# MAGIC 
# MAGIC Databricksではデータパイプラインの管理・定期実行するための機能があります(Jobs)。
# MAGIC 以下のコードをJob化してみましょう。

# COMMAND ----------

### 1. ストレージから更新ファイルだけを認識して、Deltaテーブルに追記する
df_autoloader = (
  spark.readStream.format('cloudFiles')
  .option('cloudFiles.format', 'json')
  .option('cloudFiles.maxBytesPerTrigger', '50KB')
  .schema(tweet_schema)
  .load('s3a://databricks-ktmr-s3/stocknet-dataset/tweet/raw/AAPL/*')
)

(
  df_autoloader.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/daiwt2021/job/tweet.checkpoint')
  .option('maxFilesPerTrigger', 25)
  .outputMode('append')
  .trigger(once=True) # 一度だけ処理
  .start('/tmp/daiwt2021/job/tweet.delta')
  .awaitTermination() # async => sync
)

# COMMAND ----------

### 2. 上記のDeltaテーブルからサマリのDeltaテーブルを作る

df=spark.read.format('delta').load('/tmp/daiwt2021/job/tweet.delta')

(
  df.groupBy('lang').count()
  .write.format('delta').mode('overwrite').save('/tmp/daiwt2021/job/tweet_summary.delta')
)

sql("CREATE TABLE IF NOT EXISTS tweet_summary USING delta LOCATION '/tmp/daiwt2021/job/tweet_summary.delta'")

# 確認
display(
  spark.read.format('delta').load('/tmp/daiwt2021/job/tweet_summary.delta')
)

# COMMAND ----------


