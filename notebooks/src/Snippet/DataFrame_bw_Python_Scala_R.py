# Databricks notebook source
# MAGIC %md
# MAGIC # Python, R, Scala, SQL間でのDataframeの共有方法
# MAGIC 
# MAGIC **基本方針**:
# MAGIC 基本的に、Datafarmeにtemp viewを割り当てて、各言語からそのview名でアクセスすることでDataframeを共有できます。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. サンプルのDatafarmeの準備

# COMMAND ----------

# MAGIC %python
# MAGIC 
# MAGIC df_python = (
# MAGIC   spark
# MAGIC   .read
# MAGIC   .format('csv')
# MAGIC   .option('Header', True)
# MAGIC   .option('inferSchema', True)
# MAGIC   .load('/databricks-datasets/learning-spark-v2/flights/departuredelays.csv')
# MAGIC )
# MAGIC 
# MAGIC display(df_python)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Python -> Scala, R, SQL

# COMMAND ----------

# DBTITLE 1,Python上でDataframeにtemp viewを割り当てる
# MAGIC %python
# MAGIC 
# MAGIC # daraframeにtemp viewを割り当てる
# MAGIC df_python.createOrReplaceTempView('sample_dataframe')

# COMMAND ----------

# DBTITLE 1,Scalaで受け取る
# MAGIC %scala
# MAGIC 
# MAGIC val df_scala = spark.table("sample_dataframe")
# MAGIC display(df_scala)
# MAGIC 
# MAGIC val df_scala_by_sql = spark.sql("SELECT * FROM sample_dataframe")
# MAGIC display(df_scala_by_sql)

# COMMAND ----------

# DBTITLE 1,Rで受け取る
# MAGIC %r
# MAGIC require(SparkR)
# MAGIC 
# MAGIC df_r <- sql("SELECT * FROM sample_dataframe")
# MAGIC display(df_r)

# COMMAND ----------

# DBTITLE 1,SQLで受け取る
# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM sample_dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Scala, R -> Python
# MAGIC 
# MAGIC 同様にtemp view化して、pythonから参照する

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC df_scala.createOrReplaceTempView("df_scala_tempview")

# COMMAND ----------

# MAGIC %r
# MAGIC require(SparkR)
# MAGIC 
# MAGIC createOrReplaceTempView(df_r, "df_r_tempview")

# COMMAND ----------

# MAGIC %python
# MAGIC #pythonから参照する
# MAGIC 
# MAGIC 
# MAGIC # 1. Scalaから受け取る
# MAGIC df_from_scala = spark.table('df_scala_tempview')
# MAGIC ## もしくは、以下でもOK
# MAGIC # df_from_scala = spark.sql('SELECT * FROM df_scala_tempview')
# MAGIC 
# MAGIC display( df_from_scala )
# MAGIC 
# MAGIC 
# MAGIC # 2. Rから受け取る
# MAGIC df_from_r = spark.table('df_r_tempview')
# MAGIC ## もしくは、以下でもOK
# MAGIC # df_from_r = spark.sql('SELECT * FROM df_r_tempview')
# MAGIC 
# MAGIC display( df_from_r )
