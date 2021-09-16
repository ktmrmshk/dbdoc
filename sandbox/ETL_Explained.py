# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # ETL Explained
# MAGIC 
# MAGIC * read from / write to files
# MAGIC * slice, select, filter
# MAGIC * aggregation - group by
# MAGIC * sort
# MAGIC * join
# MAGIC 
# MAGIC * subqeries
# MAGIC * UDF

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/airlines/part-00659

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC head /dbfs/databricks-datasets/airlines/part-00659

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/databricks-datasets/samples/lending_club/parquet

# COMMAND ----------

path='/databricks-datasets/samples/lending_club/parquet'
df = spark.read.format('parquet').load(path)
display(df)

# COMMAND ----------

df.count()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df_silver = (
  df
  .select('addr_state', 'loan_status', 'loan_amnt', 'int_rate')
  .withColumnRenamed('addr_state', 'state')
)

display(df_silver)

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

df2 = (
  df
#  .filter(col('int_rate') < 10.0)
#  .filter(df.int_rate < 10.0)
  .withColumn(
    'int_rate_double', 
    regexp_replace('int_rate', '%', '').cast('double')
  )
  .filter(col('int_rate_double') < 10.0)
  .select(col('addr_state').alias('state'), col('loan_status'), col('loan_amnt'), col('int_rate'), col('int_rate_double') )
)

display(df2)

# COMMAND ----------

df_airline = spark.read.format('csv').load('/databricks-datasets/airlines/*')
df_airline.count()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %fs ls /home/masahiko.kitamura@databricks.com/

# COMMAND ----------

df.write.mode('overwrite').format('delta').save('dbfs:/home/masahiko.kitamura@databricks.com/airlines')

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /dbfs/databricks-datasets/airlines/

# COMMAND ----------

# MAGIC %sh 
# MAGIC du -sh /dbfs/databricks-datasets/airlines/

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -l /dbfs/home/masahiko.kitamura@databricks.com/airlines/

# COMMAND ----------

# MAGIC %sh
# MAGIC du -sh /dbfs/home/masahiko.kitamura@databricks.com/airlines/

# COMMAND ----------

display(df)

# COMMAND ----------

df2_sum = df.groupby('issue_d').count().orderBy('issue_d')
df2_sum = df2_sum.filter(col('issue_d').contains('-'))
display(df2_sum)

# COMMAND ----------

from pyspark.sql.functions import split, month, date_format, to_timestamp, year

df3 = (
  df2_sum
  .withColumn('month_txt', split(col('issue_d'), '-').getItem(0))
#  .withColumn('year', split(col('issue_d'), '-').getItem(1))
#  .withColumn('month_num', month('month_txt'))
  .withColumn('dat', to_timestamp('issue_d', 'MMM-yyyy'))
  .withColumn('month', month('dat'))
  .withColumn('year', year('dat'))
  .select('count', 'year', 'month', 'dat', 'month_txt')
  .orderBy('dat')
)
display(df3)

# COMMAND ----------

#from pyspark.sql.functions import replace

m = {'Jan': '1', 'Feb': '2'}
df4 = (
  df3
  .replace(to_replace=m, subset=['month_txt'])
  .withColumn('month_int', col('month_txt').cast('integer'))
)

display(df4)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Lesson1 - ETL1

# COMMAND ----------

# MAGIC %fs head /mnt/training/zips.json

# COMMAND ----------

df_j = spark.read.format('json').load('/mnt/training/zips.json')
df_j.printSchema()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

schema1 = StructType([
  StructField('city', StringType(), True),
  StructField('pop', IntegerType(), True)
])

df_schema1 = spark.read.format('json').schema(schema1).load('/mnt/training/zips.json')
df_schema1.printSchema()

# COMMAND ----------

schema2='city string, pop int'
df_schema2 = spark.read.format('json').schema(schema2).load('/mnt/training/zips.json')
df_schema2.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## dig nested data

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType

schema3 = StructType([
  StructField('city', StringType(), True),
  StructField('loc', ArrayType(FloatType(), True), True),
  StructField('pop', IntegerType(), True)
])

df_schema3 = spark.read.format('json').schema(schema3).load('/mnt/training/zips.json')
df_schema3.printSchema()

display(df_schema3)

# COMMAND ----------

from pyspark.sql.functions import expr

display( 
  df_schema3
  .select(
    col('loc')[0].alias('loc0'),
    col('loc')[1].alias('loc1')
  )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Handling Corrupt data
# MAGIC 
# MAGIC Currupt data may be
# MAGIC 
# MAGIC * missing value
# MAGIC * Schema mismatch
# MAGIC * Format mismatch(capital, number digit, datetime format etc)
# MAGIC * Incomplete records

# COMMAND ----------

# generating sample data

data = [] #'{"a": 1, "b":2, "c":3}|{"a": 1, "b":2, "c":3}|{"a": 1, "b, "c":10}'.split('|')
data.append('{"a": 1, "b":2, "c":3}')
data.append('{"a": 1, "b":2, "c":3}')
data.append('{"a": 1, "b, "c":10}')

rdd = sc.parallelize(data)
rdd.take(5)

# COMMAND ----------

# PERMISSIVE

brokenDF = (
  spark.read
  .option('mode', 'PERMISSIVE')
  .option('columnNameOfCorruptRecord', '_corrupt_rec')
  .json(rdd)
)

display(brokenDF)

# COMMAND ----------

# DROP

brokenDF = (
  spark.read
  .option('mode', 'DROPMALFORMED')
#  .format('json')
#  .load(rdd)
  .json(rdd)
)

display(brokenDF)

# COMMAND ----------

# Fail fast

try:
  brokenDF = (
    spark.read
    .option('mode', 'FAILFAST')
    .json(rdd)
  )
except Exception as e:
  print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC # Parquet

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -lh /dbfs/mnt/training/Chicago-Crimes-2018.csv

# COMMAND ----------

# MAGIC %sh
# MAGIC head 3 /dbfs/mnt/training/Chicago-Crimes-2018.csv

# COMMAND ----------

crimeDF = (
  spark.read
  .format('csv')
  .option('header', True)
  .option('delimiter', '\t')
#  .option('timestampFormat', 'MM/dd/yyyy HH:mm:ss a')
  .option("timestampFormat", "MM/dd/yyyy hh:mm:ss a")
  .option('inferSchema', True)
  .load('/mnt/training/Chicago-Crimes-2018.csv')
)

crimeDF.printSchema()

display(crimeDF)

# COMMAND ----------

#from pyspark.sql.functions import to_timestamp
#
#display(
#  crimeDF
#  .withColumn('dat', to_timestamp('Date', 'MM/dd/yyyy hh:mm:ss a'))
#  .select('Date', 'dat')
#)

# COMMAND ----------

# eliminate space in col name

colnames = crimeDF.columns
renames = [ c.replace(' ', '_') for c in colnames ]

renamedDF = (
   crimeDF.toDF(*renames)
)

display(renamedDF)

# COMMAND ----------

# write to parquet with 10 partitions

(
  renamedDF.repartition(10).write
  .mode('overwrite')
  .format('parquet')
  .save('/home/masahiko.kitamura@databricks.com/crimeRepartitioned.parquet')
)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lh /dbfs/home/masahiko.kitamura@databricks.com/crimeRepartitioned.parquet/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Lesson2
# MAGIC 
# MAGIC ## Transformation
# MAGIC 
# MAGIC * Normalizing values
# MAGIC * Imputing null or missing values
# MAGIC * Deduplicating data
# MAGIC * Perf data base rollups
# MAGIC * pivot/expliding data

# COMMAND ----------

df = spark.range(1000,10000)
display(df)

# COMMAND ----------

df.first()[0]

# COMMAND ----------

from pyspark.sql.functions import col, max, min

colMin = df.select(min('id')).first()[0]
colMax = df.select(max('id')).first()[0]

normDF = (
  df
  .withColumn('normed', (col('id') - colMin) / (colMax - colMin) )
)

display(normDF)

# COMMAND ----------

# imputer

brokenDF = spark.createDataFrame([
  (11, 65, 3),
  (12, 53, None),
  (1, None, 3),
  (2, 32, 11)],
  ['hour', 'temp', 'wind']
)

display(brokenDF)

# COMMAND ----------

# Drop nulls

droppedDF = brokenDF.dropna('any')

display(droppedDF)

# COMMAND ----------

# impute with values

display(
  brokenDF.na.fill({'temp':111111, 'wind': 2222222})
)

# COMMAND ----------

# deduplicate

dupDF = spark.createDataFrame([
  (123, 'Conor', 'red'),
  (123, 'conor', 'red'),
  (123, 'Dorothy', 'blue'),
  (321, 'Doug', 'aqua')],
  ['id', 'name','color']
)

display(dupDF)

# COMMAND ----------

display( 
  dupDF.dropDuplicates(['id', 'color'])
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## homework: explode, pivot, cube, rollup

# COMMAND ----------

# MAGIC %md 
# MAGIC # UDF

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## pyspark UDF

# COMMAND ----------

def myfunc(x):
  return x.upper()

myfunc('abcEfG')

# COMMAND ----------

from pyspark.sql.types import StringType

myfuncUDF = spark.udf.register('myfuncSQLUDF', myfunc, StringType())

# COMMAND ----------

from pyspark.sql.functions import sha1, rand
randomDF = (
  spark.range(1, 10000 * 10 * 10 * 10)
  .withColumn('random_value', rand(seed=10).cast('string'))
  .withColumn('hash', sha1('random_value'))
  .drop('random_value')
)

display(randomDF)

# COMMAND ----------

# apply UDF

display(
  randomDF
  .select('*', myfuncUDF('hash').alias('udfed'))
)

# COMMAND ----------

randomDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## UDF in SQL

# COMMAND ----------

randomDF.createOrReplaceTempView('randomTable')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT id, hash, myfuncSQLUDF(hash) as udfed from randomTable;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## UDF Performance

# COMMAND ----------

from pyspark.sql.functions import col, rand

randomFloatDF = (
  spark.range(0, 1e8)
  .withColumn('id', ( col('id') / 1e3 ).cast('integer'))
  .withColumn('random_float', rand() )
)

randomFloatDF.cache()
print( randomFloatDF.count() )

display(randomFloatDF)


# COMMAND ----------

# plus_one UDF

from pyspark.sql.types import FloatType

plusOneUDF = spark.udf.register('plusOneUDF', lambda x: x+1, FloatType())

# COMMAND ----------

# MAGIC %timeit randomFloatDF.withColumn('inc_float', plusOneUDF('random_float') ).count()

# COMMAND ----------

# MAGIC %timeit randomFloatDF.withColumn('inc_float', col('random_float')+1 ).count()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Advanced UDF

# COMMAND ----------

# MAGIC %md
# MAGIC ## multiple cols UDF

# COMMAND ----------

def myAdd(x, y):
  return x+y

from pyspark.sql.types import IntegerType

myAddUDF = spark.udf.register('myAddUDF', myAdd, IntegerType())

# COMMAND ----------

intDF = spark.createDataFrame([
  (1,2),
  (3,4),
  (5,6)
],['col1', 'col2'])

display(intDF)

# COMMAND ----------

display(
  intDF
  .withColumn('added', myAddUDF('col1', 'col2'))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## complex(returned multi cols)

# COMMAND ----------

from pyspark.sql.types import FloatType, IntegerType, DoubleType, StructType, StructField

ret_schema = StructType([
  StructField('plus', DoubleType(), True),
  StructField('multi', DoubleType(), True),
  StructField('div', DoubleType(), True)  
])

def myCalc(x, y):
  return (float(x+y), float(x*y), float(x/y))

myCalc(1,2)

# COMMAND ----------

from pyspark.sql.functions import col

myCalcUDF = spark.udf.register('myCalcUDF', myCalc, ret_schema)

display(
  intDF.select('*', myCalcUDF('col1', 'col2').alias('calc') )
  .withColumn('plus', col('calc')['plus'])
  .withColumn('multi', col('calc')['multi'])
  .withColumn('div', col('calc')['div'])
)

# COMMAND ----------

col('foobar') + 1

# COMMAND ----------

# MAGIC %md
# MAGIC # join

# COMMAND ----------

df = spark.read.format('parquet').load('/mnt/training/day-of-week')
display(df)

# COMMAND ----------

from pyspark.sql.functions import col, date_format

pageDF = (
  spark.read
  .format('parquet')
  .load('/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet/')
#  .withColumn( 'dow', date_format( col('timestamp'), 'u' ))
  .withColumn("dow", date_format(col("timestamp"), "F").alias("dow"))
)

display(pageDF)

# COMMAND ----------

# normal join
joinedDF = pageDF.join(df, 'dow')
display(joinedDF)

# COMMAND ----------

# agg
from pyspark.sql.functions import col
aggDF = (
  joinedDF
  .groupby( col('dow'), col('longName'), col('abbreviated'), col('shortName')  )
  .sum('requests')
  .withColumnRenamed('sum(requests)', 'Requests')
  .orderBy(col('dow'))
)

display(aggDF)

# COMMAND ----------

aggDF.explain()

# COMMAND ----------

# threshold size of Broadcast join

spark.conf.get('spark.sql.autoBroadcastJoinThreshold')

# COMMAND ----------

# disable broadcast join
spark.conf.set('spark.sql.autoBroadcastJoinThreshold', -1)
spark.conf.get('spark.sql.autoBroadcastJoinThreshold')

# COMMAND ----------

aggDF.explain()

# COMMAND ----------

# set to dafault

spark.conf.set('spark.sql.autoBroadcastJoinThreshold', 10485760)
spark.conf.get('spark.sql.autoBroadcastJoinThreshold')

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -lh /dbfs/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet

# COMMAND ----------

pageDF = spark.read.parquet('/mnt/training/wikipedia/pageviews/pageviews_by_second.parquet')
display(pageDF)
print(pageDF.count())

# COMMAND ----------

partition = pageDF.rdd.getNumPartitions()
print(partition)

# COMMAND ----------

rePageDF = pageDF.repartition(16)
partition = rePageDF.rdd.getNumPartitions()
print(partition)

# COMMAND ----------

# reduce partition
coPageDF = rePageDF.coalesce(2)
partition = coPageDF.rdd.getNumPartitions()
print(partition)

# COMMAND ----------

# default partition
spark.conf.get('spark.sql.shuffle.partitions')


# COMMAND ----------

# write to file
coPageDF.write.mode('overwrite').format('parquet').save('/tmp/masahiko.kitamura@databricks.com/pages')

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/tmp/masahiko.kitamura@databricks.com/pages

# COMMAND ----------

# MAGIC %md
# MAGIC # Widgets

# COMMAND ----------

# dropdown
dbutils.widgets.dropdown('MyDropdown', '1', ['1','2','3'])

# text
dbutils.widgets.text('mytest', '1')

# combobox
dbutils.widgets.combobox('myCombo', 'a', ['a', 'b', 'c'] )

# multiselect
dbutils.widgets.multiselect('myMulti', '1', ['1', '2', '3'])

# COMMAND ----------

mytest_val = dbutils.widgets.get('mytest')
print(mytest_val)

# COMMAND ----------


