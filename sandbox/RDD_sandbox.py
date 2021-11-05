# Databricks notebook source
sc

# COMMAND ----------

data = range(1000)
data


# COMMAND ----------

rdd = sc.parallelize(data,4)

# COMMAND ----------

rdd.getNumPartitions()

# COMMAND ----------

rdd2 = rdd.reduce(lambda a, b: a+b)

# COMMAND ----------

rdd_txt = sc.textFile('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv')

# COMMAND ----------

rdd_txt.map(lambda a: len(a)).reduce(lambda a, b: a+b)

# COMMAND ----------

rdd_txt.map(lambda a: len(a)).explain()

# COMMAND ----------

spark.read.option("header", True).csv('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv').select("id").explain()

# COMMAND ----------

lines = sc.textFile('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv').map(lambda x: x.replace('a', 'A'))
lines.take(100)

# COMMAND ----------

lines.persist()

# COMMAND ----------

lines.take(1000)

# COMMAND ----------


