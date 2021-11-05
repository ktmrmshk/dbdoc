# Databricks notebook source
# MAGIC %md
# MAGIC # CSVファイルのロード

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/nyctaxi/tripdata/yellow

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/nyctaxi

# COMMAND ----------

# MAGIC %sh  zcat /dbfs/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2009-01.csv.gz | head

# COMMAND ----------

df = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .load('/databricks-datasets/nyctaxi/tripdata/yellow/*.csv.gz')
)

display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

spark.sql("set spark.databricks.delta.autoCompact.enabled = true")


# COMMAND ----------

username = 'masahiko.kitamura@databricks.com'
df.write.format('delta').mode('overwrite').save(f'/tmp/{username}/nytaxi.delta')

# COMMAND ----------

df = spark.read.format('delta').load(f'/tmp/{username}/nytaxi.delta')
df.count()

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

display( df.where( df.payment_type == 'CASH') )

# COMMAND ----------


