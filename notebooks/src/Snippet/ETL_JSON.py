# Databricks notebook source
# MAGIC %md
# MAGIC # JSONをParseしてフラットにする
# MAGIC 
# MAGIC 対象のJSONデータ(`example.json`)
# MAGIC 
# MAGIC ```
# MAGIC {
# MAGIC     "member": "038408372",
# MAGIC     "store": "64801",
# MAGIC     "date": "2020-05-13 11:09",
# MAGIC     "total": 1222,
# MAGIC     "receipt_no": "000105",
# MAGIC     "lines": [
# MAGIC         {
# MAGIC             "name": "商品番号00007663番",
# MAGIC             "price": 128,
# MAGIC             "count": 1,
# MAGIC             "jancode": "4996124621169"
# MAGIC         },
# MAGIC         {
# MAGIC             "name": "商品番号00002830番",
# MAGIC             "price": 123,
# MAGIC             "count": 1,
# MAGIC             "jancode": "4957240178104"
# MAGIC         },
# MAGIC         {
# MAGIC             "name": "商品番号00003513番",
# MAGIC             "price": 85,
# MAGIC             "count": 1,
# MAGIC             "jancode": "4934075919210"
# MAGIC         },
# MAGIC         {
# MAGIC             "name": "商品番号00008037番",
# MAGIC             "price": 886,
# MAGIC             "count": 1,
# MAGIC             "jancode": "4902421435385"
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC ```
# MAGIC 
# MAGIC 
# MAGIC が3行あるデータ
# MAGIC ```
# MAGIC {"member":"038408372","store":"64801","date":"2020-05-13 11:09","total":1222,"receipt_no":"000105","lines":[{"name":"商品番号00007663番","price":128,"count":1,"jancode":"4996124621169"},{"name":"商品番号00002830番","price":123,"count":1,"jancode":"4957240178104"},{"name":"商品番号00003513番","price":85,"count":1,"jancode":"4934075919210"},{"name":"商品番号00008037番","price":886,"count":1,"jancode":"4902421435385"}]}
# MAGIC {"member":"038408373","store":"64801","date":"2020-05-14 11:09","total":1222,"receipt_no":"000105","lines":[{"name":"商品番号00007663番","price":128,"count":1,"jancode":"4996124621169"},{"name":"商品番号00002830番","price":123,"count":1,"jancode":"4957240178104"},{"name":"商品番号00003513番","price":85,"count":1,"jancode":"4934075919210"},{"name":"商品番号00008037番","price":886,"count":1,"jancode":"4902421435385"}]}
# MAGIC {"member":"038408374","store":"64801","date":"2020-05-15 11:09","total":1222,"receipt_no":"000105","lines":[{"name":"商品番号00007663番","price":128,"count":1,"jancode":"4996124621169"},{"name":"商品番号00002830番","price":123,"count":1,"jancode":"4957240178104"},{"name":"商品番号00003513番","price":85,"count":1,"jancode":"4934075919210"},{"name":"商品番号00008037番","price":886,"count":1,"jancode":"4902421435385"}]}
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,step1: まずはjsonをそのままDataFrameに読み込む
df = (
  spark.read
  .format('json')
  .load('/PATH/TO/example.json')
)

display(df)

# COMMAND ----------

# DBTITLE 1,step2: List型のカラムを行に展開する(explode)
from pyspark.sql.functions import *

# explode()はListの各要素を行に展開する。例えばlength=3のリストは3つの行に展開される
df_exploded = (
  df.withColumn( 'lines_exploded', explode('lines') )
)

display(df_exploded)

# COMMAND ----------

# DBTITLE 1,step3: JSON形式のカラムをgetItem()を使って分離する
df_flat = (
  df_exploded
  .withColumn('count', col('lines_exploded').getItem('count'))
  .withColumn('jancode', col('lines_exploded').getItem('jancode'))
  .withColumn('name', col('lines_exploded').getItem('name'))
  .withColumn('price', col('lines_exploded').getItem('price'))
  .select('date', 'count', 'jancode', 'name', 'price', 'member', 'receipt_no', 'store', 'total')
)

display(df_flat)

# COMMAND ----------

# DBTITLE 1,(補足) 上記のstep1-3をまとめて書くとこうなります
from pyspark.sql.functions import *

df_onetime = (
  spark.read
  .format('json')
  .load('/PATH/TO/example.json')

  .withColumn( 'lines_exploded', explode('lines') )
  
  .withColumn('count', col('lines_exploded').getItem('count'))
  .withColumn('jancode', col('lines_exploded').getItem('jancode'))
  .withColumn('name', col('lines_exploded').getItem('name'))
  .withColumn('price', col('lines_exploded').getItem('price'))
  .select('date', 'count', 'jancode', 'name', 'price', 'member', 'receipt_no', 'store', 'total')
)

display(df_onetime)
