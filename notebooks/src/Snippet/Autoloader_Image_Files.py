# Databricks notebook source
# MAGIC %md ## Autoloaderを使用して、Object Storageにアップロードされたファイル(新規に追加されたファイルのみ)をロードする

# COMMAND ----------

# MAGIC %md ### A: Object Storage上から読み込み、処理を実施して、結果をDelta Lakeに入れる。
# MAGIC 
# MAGIC 途中で、Rawデータ様のDeltaテーブルを作成しないパターン

# COMMAND ----------

# clean for retry
dbutils.fs.rm('/tmp/masahiko.kitamura@databricks.com/', True)

# COMMAND ----------

# 初期化設定
sql('set spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;')
sql('set spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;')

# UDFを定義(PILを使用して画像ファイルを開き、属性情報を抽出し、返す)
@udf("width long, height long, mode string")
def proc_image(image_binary):
  import io
  from PIL import Image
  f=io.BytesIO(image_binary)
  im = Image.open(f)
  return (im.width, im.height, im.mode)



# Auto Loaderで読み込む (新規ファイルのみ読み込まれる)
df_autoloader = (
  spark.readStream.format('cloudFiles')
  .option('cloudFiles.format', 'binaryFile')
  .load('s3a://databricks-ktmr-s3/sample/jpg/*.jpg')
)

# 画像処理(上記で定義したUDFを適用する)
df_processed = (
  df_autoloader
  .withColumn('added_cols', proc_image('content')) # UDFの適用
  .select(df_autoloader.columns + ['added_cols.*']) # カラムのフラット化
)

# DeltaLakeに書き出す
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
(
  df_processed.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/masahiko.kitamura@databricks.com/image.checkpoint')
  .outputMode('append')
  .trigger(once=True) # 一度だけ処理
  .start('/tmp/masahiko.kitamura@databricks.com/image.delta')
  .awaitTermination() # async => sync
)
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")


# COMMAND ----------

# 上記で書き出したDeltalakeの内容を確認
df = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/image.delta')
display( df )

# COMMAND ----------

# pandasなどで利用する (10レコードのみ)

pandas_df = df.limit(10).toPandas()
pandas_df

# COMMAND ----------

# spark Dataframeから一つのカラムを抜き出して、プレビューする
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
import io


image_binary = df.filter('path = "s3a://databricks-ktmr-s3/sample/jpg/img200kb_0014.jpg"').collect()[0]['content']
image_pil = Image.open(io.BytesIO(image_binary))
imshow(np.asarray(image_pil))

# COMMAND ----------

# MAGIC %md ### B: Object Storage上から読み込み、一度Rawテーブル化を経由して、処理結果をDelta Lakeに入れる。
# MAGIC 
# MAGIC 途中で、Rawデータ様のDeltaテーブルを作成するパターン。以下の2つのDelta lakeテーブルが作られる。
# MAGIC 
# MAGIC -------
# MAGIC 
# MAGIC 1. Rawテーブル(生データのテーブル): `raw_data.delta`
# MAGIC 2. 画像処理結果のみのテーブル: `image_processed_result.delta`
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC Rawテーブルが作られるので(生データの永続化がDeltalake内に閉じる)、他の用途(パイプラインの分岐)などに対応しやすくなる。

# COMMAND ----------

# DBTITLE 1,Object Storage -> Raw_data table
# Auto Loaderで読み込む (新規ファイルのみ読み込まれる)
df_autoloader = (
  spark.readStream.format('cloudFiles')
  .option('cloudFiles.format', 'binaryFile')
  .load('s3a://databricks-ktmr-s3/sample/jpg/*.jpg')
)

# DeltaLakeに書き出す
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
(
  df_autoloader.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/masahiko.kitamura@databricks.com/raw_data.checkpoint')
  .outputMode('append')
  .trigger(once=True) # 一度だけ処理
  .start('/tmp/masahiko.kitamura@databricks.com/raw_data.delta')
  .awaitTermination() # async => sync
)
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

# COMMAND ----------

# raw_dataテーブルの結果確認

display(
  spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/raw_data.delta')
)

# COMMAND ----------

# DBTITLE 1,Raw_data table -> image_processed_result table
# [再掲]
# UDFを定義(PILを使用して画像ファイルを開き、属性情報を抽出し、返す)
@udf("width long, height long, mode string")
def proc_image(image_binary):
  import io
  from PIL import Image
  f=io.BytesIO(image_binary)
  im = Image.open(f)
  return (im.width, im.height, im.mode)



# delta lakeからstreamingで読み込む (新規ファイルのみ読み込まれる)
df_raw = (
  spark.readStream.format('delta')
  .load('/tmp/masahiko.kitamura@databricks.com/raw_data.delta')
)


# 画像処理(上記で定義したUDFを適用する)
df_processed = (
  df_raw
  .withColumn('added_cols', proc_image('content')) # UDFの適用
  .select(['path','added_cols.*']) # カラムのフラット化(`path`を外部キーにする想定)
)


# DeltaLakeに書き出す
(
  df_processed.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/masahiko.kitamura@databricks.com/image_processed_result.checkpoint')
  .outputMode('append')
  .trigger(once=True) # 一度だけ処理
  .start('/tmp/masahiko.kitamura@databricks.com/image_processed_result.delta')
  .awaitTermination() # async => sync
)

# COMMAND ----------

#結果の確認

display(
  spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/image_processed_result.delta')
)

# COMMAND ----------

# rawテーブルとのjoinで、パターンAと同じ結果を得る(python版)

df_raw                    = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/raw_data.delta')
df_image_processed_result = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/image_processed_result.delta')

df_joined = (
  df_raw.join( df_image_processed_result, df_raw.path == df_image_processed_result.path)
)

display(df_joined)


# COMMAND ----------

# rawテーブルとのjoinで、パターンAと同じ結果を得る(SQL版)

df_raw                    = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/raw_data.delta')
df_image_processed_result = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/image_processed_result.delta')

df_raw.createOrReplaceTempView('table_raw')
df_image_processed_result.createOrReplaceTempView('table_image_processed_result')

df_joined = spark.sql('''
  SELECT * FROM table_raw r
  JOIN table_image_processed_result i
  WHERE r.path = i.path
''')

display( df_joined )

# COMMAND ----------

# MAGIC %sql
# MAGIC select path from table_image_processed_result

# COMMAND ----------

# MAGIC %md ## Performance Test: image in delta vs image file

# COMMAND ----------

# MAGIC %md ### 1. from file each (file path)

# COMMAND ----------

img_filelist=[]
img_filelist += dbutils.fs.ls('/databricks-datasets/flower_photos/daisy/')
img_filelist += dbutils.fs.ls('/databricks-datasets/flower_photos/dandelion/')
img_filelist += dbutils.fs.ls('/databricks-datasets/flower_photos/roses/')
img_filelist += dbutils.fs.ls('/databricks-datasets/flower_photos/sunflowers/')
img_filelist += dbutils.fs.ls('/databricks-datasets/flower_photos/tulips/')
df = spark.createDataFrame(img_filelist)
display(df)
df.count()

# COMMAND ----------

@udf("width long, height long, mode string")
def proc_image_from_path(image_path):
  from PIL import Image
  image_path = image_path.replace('dbfs:', '/dbfs')
  im = Image.open(image_path)
  return (im.width, im.height, im.mode)

# COMMAND ----------

display(
df.withColumn('udf_ret', proc_image_from_path('path') )
)

# COMMAND ----------

# MAGIC %md ### 2. from delta (image data in delta)

# COMMAND ----------

dbutils.fs.rm('/tmp/masahiko.kitamura@databricks.com/perftest', True)

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
(
  spark
  .read.format('binaryFile').load('/databricks-datasets/flower_photos/*/*.jpg')
  .write.format('delta').save('/tmp/masahiko.kitamura@databricks.com/perftest/images.delta')
)
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(1) from delta.`/tmp/masahiko.kitamura@databricks.com/perftest/images.delta`;

# COMMAND ----------

# UDFを定義(PILを使用して画像ファイルを開き、属性情報を抽出し、返す)
@udf("width long, height long, mode string")
def proc_image_from_delta(image_binary):
  import io
  from PIL import Image
  f=io.BytesIO(image_binary)
  im = Image.open(f)
  return (im.width, im.height, im.mode)

# COMMAND ----------

df_delta = spark.read.format('delta').load('/tmp/masahiko.kitamura@databricks.com/perftest/images.delta')

display(
  df_delta.withColumn('udf_ret', proc_image_from_delta('content') )
)

# COMMAND ----------

# MAGIC %md Optional) Optimize for file compaction

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.sql.parquet.compression.codec = uncompressed;
# MAGIC OPTIMIZE delta.`/tmp/masahiko.kitamura@databricks.com/perftest/images.delta`;
# MAGIC SET spark.databricks.delta.retentionDurationCheck.enabled = false;
# MAGIC VACUUM "/tmp/masahiko.kitamura@databricks.com/perftest/images.delta" RETAIN 0 HOURS;
# MAGIC SET spark.databricks.delta.retentionDurationCheck.enabled = true;
# MAGIC SET spark.sql.parquet.compression.codec = snappy;
