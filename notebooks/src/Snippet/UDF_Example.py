# Databricks notebook source
# MAGIC %md # Sparkのユーザー定義関数(UDF)の使い方

# COMMAND ----------

# MAGIC %md # 1. 通常のUDF

# COMMAND ----------

schema_info='''
  InvoiceNo string,
  StockCode string,
  Description string,
  Quantity int,
  InvoiceDate string,
  UnitPrice double,
  CustomerID string,
  Country string
'''

df_raw = (
  spark
  .read
  .format('csv')
  .schema(schema_info)
  .option('Header', True)
  .load('/databricks-datasets/online_retail/data-001/data.csv')
)

display(df_raw)

df_raw.createOrReplaceTempView('df_raw_view')

# COMMAND ----------

# DBTITLE 1,例1) 一番単純なUDF 
from pyspark.sql.functions import udf, col

# Descriptionカラムを小文字にするUDFを定義する

@udf('string')
def to_lower(s):
  if s is not None: # Nullではない場合
    return s.lower()
  return s

# UDFを適用する

df_lower = (
  df_raw
  .withColumn('Lower Description', to_lower('Description') )
)

display(df_lower)

# COMMAND ----------

# SQLでもUDFを使う場合はRegisterする必要がある。

# ここでは、先ほど定義した`to_lower()`を`TO_LOWER_UDF`と言う名前で登録する。
spark.udf.register("TO_LOWER_UDF", to_lower)

# COMMAND ----------

# MAGIC %sql
# MAGIC --- 上記のUDFをSQL上で適用させる(結果はpython時と同じ)
# MAGIC SELECT *, TO_LOWER_UDF(Description) as Lower_Description FROM df_raw_view

# COMMAND ----------

# DBTITLE 1,例2) 複数カラムを引数にする場合
@udf('string', 'string')
def make_summary_description(desc, country):
  ret = f'{desc} sold in {country}'
  return ret

df_desc = (
  df_raw
  .withColumn('Summary Desc', make_summary_description('Description', 'Country') )
)

display(df_desc)

# COMMAND ----------

# MAGIC %md # 2. Pandas UDF(Vectorized UDF)
# MAGIC 
# MAGIC Apache Arrowを使用して従来のUDFに比べて高速に実行可能。詳しくは[こちらのドキュメント](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/spark-sql/udf-python-pandas)を参照ください。

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType

# UDFを定義する
def multiply_func(a: pd.Series, b: pd.Series) -> pd.Series:
  return a*b


# pandasコードで関数の動作を確認しておく
x = pd.Series([1, 12, 123])
y = pd.Series([10, 100, 1000])
print( multiply_func(x, y))

print('-----------------')

# SparkDFで使えるようにpandas_udf化する
multiply = pandas_udf(multiply_func, returnType=DoubleType())

# Spark DFに適用する( 売り上げ = Quantity * UnitPrice を計算する)
df_sales = (
  df_raw.withColumn('sales', multiply( col('Quantity'), col('UnitPrice') ) )
)

# dataframeを確認
display(df_sales)

# COMMAND ----------

#SQLで使うために登録する

spark.udf.register('multiply_udf', multiply)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *, multiply_udf(Quantity, UnitPrice) as Sales FROM df_raw_view
