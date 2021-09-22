# Databricks notebook source
# MAGIC %md
# MAGIC # 非構造化データ(画像)のDeltaでの取り扱い
# MAGIC 
# MAGIC 1. imageフォーマットでDeltaテーブルに入れる
# MAGIC 1. binaryフォーマットでDeltaテーブルに入れる
# MAGIC 1. ファイルパスのみをDeltaテーブルに入れる

# COMMAND ----------

sample_img_dir = "/databricks-datasets/cctvVideos/train_images/"
display( dbutils.fs.ls(sample_img_dir) )

# COMMAND ----------

df = spark.read.format('image').load(sample_img_dir)
display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(
  df.select('image.*', 'label')

)

# COMMAND ----------

# MAGIC %md binary format

# COMMAND ----------

df = (
  spark
  .read
  .format('binaryFile')
  .option('mimeType', 'image/*')
  .load('/databricks-datasets/cctvVideos/train_images/label=0/*.jpg')
)
  
display(df) 

# COMMAND ----------

spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

(
  df.write
  .format('delta')
  .save('s3://databricks-ktmr-s3/image_delta/example123.delta')
)




spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
#spark.conf.get("spark.sql.parquet.compression.codec") # <= 'snappy'

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/cctvVideos/train_images/label=0/

# COMMAND ----------

from pyspark.sql.functions import col, sum
display( df.agg(sum(col('length'))) )

# COMMAND ----------

display(
  dbutils.fs.ls('s3://databricks-ktmr-s3/image_delta/example123.delta/') 
)

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists ktmrtbl
# MAGIC using delta
# MAGIC location 's3://databricks-ktmr-s3/image_delta/example123.delta/'

# COMMAND ----------

# MAGIC %sql 
# MAGIC optimize ktmrtbl

# COMMAND ----------

# MAGIC %sql
# MAGIC set spark.databricks.delta.retentionDurationCheck.enabled = false

# COMMAND ----------

# MAGIC %sql
# MAGIC vacuum ktmrtbl retain 0 hours

# COMMAND ----------


