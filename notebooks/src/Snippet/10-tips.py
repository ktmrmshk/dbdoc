# Databricks notebook source
df = spark.read.csv('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
df.show(5)

# COMMAND ----------

# MAGIC %fs head dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv

# COMMAND ----------

df = spark.read.csv('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv', header=True)
df.show(5)

# COMMAND ----------

df = spark.read.format('csv').option('header', True).load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
df.show(5)

# COMMAND ----------

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.show(5)

# COMMAND ----------

# MAGIC %md schemaを見てみる

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_pre = (
  spark.read
  .format('csv')
  .option('header', True)
  .option('inferSchema', True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df_pre.printSchema()

# COMMAND ----------

schema_DDLformat = '''
_c0 Integer,
carat DOUBLE,
cut String,
color String,
clarity String,
depth DOUBLE,
table Integer,
price Integer,
x DOUBLE,
y DOUBLE,
z DOUBLE
'''

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .schema(schema_DDLformat)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.printSchema()

# COMMAND ----------

from pyspark.sql.types import *

schema_StructType = StructType([
  StructField('_c0', IntegerType(), True),
  StructField('carat', DoubleType(), True),
  StructField('cut', StringType(), True),
  StructField('color', StringType(), True),
  StructField('clarity', StringType(), True),
  StructField('depth', DoubleType(), True),
  StructField('table', IntegerType(), True),
  StructField('price', IntegerType(), True),
  StructField('x', DoubleType(), True),
  StructField('y', DoubleType(), True),
  StructField('z', DoubleType(), True)
])

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .schema(schema_StructType)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df.printSchema()

# COMMAND ----------

for f in df.schema:
  print(f)

# COMMAND ----------

df_pre = (
  spark.read
  .format('csv')
  .option('header', True)
  .option('inferSchema', True)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

df_pre.printSchema()

# COMMAND ----------

df.summary().show()

# COMMAND ----------

df.summary("count", "min", "25%", "50%", "75%", "max").show()

# COMMAND ----------

display(df.summary() )




# COMMAND ----------

df.summary().collect()[1]['table']

# COMMAND ----------

table_mean   = df.summary().collect()[1]['table']
table_stddev = df.summary().collect()[2]['table']
table_median = df.summary().collect()[5]['table']

print(f'mean   => {table_mean}')
print(f'stddev => {table_stddev}')
print(f'median => {table_median}')

# COMMAND ----------

df.where('table is null').count()

# COMMAND ----------

df.where('table is null').show(5)

# COMMAND ----------

# MAGIC %md nullところを埋める

# COMMAND ----------



# COMMAND ----------

# MAGIC %md 値を取り出す

# COMMAND ----------

# MAGIC %md データフレームを永続化させておきたい

# COMMAND ----------

from pyspark.sql.types import *

schema_StructType = StructType([
  StructField('_c0', IntegerType(), True),
  StructField('carat', DoubleType(), True),
  StructField('cut', StringType(), True),
  StructField('color', StringType(), True),
  StructField('clarity', StringType(), True),
  StructField('depth', DoubleType(), True),
  StructField('table', IntegerType(), True),
  StructField('price', IntegerType(), True),
  StructField('x', DoubleType(), True),
  StructField('y', DoubleType(), True),
  StructField('z', DoubleType(), True)
])

df = (
  spark.read
  .format('csv')
  .option('header',True)
  .schema(schema_StructType)
  .load('dbfs:/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)


# COMMAND ----------

df_dropped = df.dropna()
df_dropped.summary('count').show()


# COMMAND ----------

df_dropped.where('table is null').count()

# COMMAND ----------

int( table_median )

# COMMAND ----------

df_imputed = df.na.fill(  int(table_median), 'table')

# COMMAND ----------

df_imputed.summary('count').show()

# COMMAND ----------

df.where('table is null').limit(3).show()

# COMMAND ----------

df_imputed.where('_c0 in (67, 178, 185)').show()

# COMMAND ----------

df

# COMMAND ----------

pandas_df = df.toPandas()
print( pandas_df.head() )

# COMMAND ----------

spark_df = spark.createDataFrame(pandas_df)
spark_df.show(5)

# COMMAND ----------


