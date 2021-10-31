# Databricks notebook source
# MAGIC %md # CSVファイルの読み込み、データの前処理例の例
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC 以下の処理のサンプルコードです。
# MAGIC 
# MAGIC 
# MAGIC ```
# MAGIC [file]
# MAGIC diamonds.csv
# MAGIC  |
# MAGIC  | (file load)
# MAGIC  v
# MAGIC [spark dataframe]
# MAGIC df_raw
# MAGIC  |
# MAGIC  | (remove outlier vals)
# MAGIC  v
# MAGIC [spark dataframe]          [file]
# MAGIC df_cleaned                 color_desk.csv
# MAGIC  |                          |
# MAGIC  | (categolize by price)    | (file load)
# MAGIC  v                          v
# MAGIC [spark dataframe]          [spark dataframe]
# MAGIC df_proce_categolized       df_color_desc
# MAGIC  |                          |
# MAGIC  +------------+-------------+
# MAGIC               | (join by color)
# MAGIC               v
# MAGIC              [spark dataframe]
# MAGIC              df_joined
# MAGIC               |
# MAGIC               | (write to storage with delta format)
# MAGIC               v
# MAGIC              [delta file]
# MAGIC              df_joined.delta
# MAGIC               |
# MAGIC               | (load from delta file)
# MAGIC               v
# MAGIC              [spark dataframe]
# MAGIC              df_joined_delta
# MAGIC               |
# MAGIC               | (retrieve by conditions)
# MAGIC               v
# MAGIC              [spark dataframe]
# MAGIC              df_filtered
# MAGIC               |
# MAGIC               | toPandas()
# MAGIC               v
# MAGIC              [pandas dataframe]
# MAGIC              df_filtered_pandas
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,前処理対象のCSVファイル
# MAGIC 
# MAGIC %fs head '/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv's

# COMMAND ----------


# 読み込むCSVのスキーマを設定する
diamonds_schema = '''
  id int,
  carat double,
  cut string,
  color string,
  clarity string,
  depth double,
  table double,
  price double,
  x double,
  y double,
  z double
'''

# CSVをSpark DataFrameとして読み込む
df_raw = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .schema(diamonds_schema)
  .load('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
)

# DataFrameの表示
display( df_raw )

# DataFrameのスキーマの確認
df_raw.printSchema()

# COMMAND ----------

# DBTITLE 1,(補足) 統計値を確認する
dbutils.data.summarize( df_raw )

# COMMAND ----------

# DBTITLE 1,depthとxについて平均、標準偏差を計算する
from pyspark.sql.functions import col, mean, stddev, lit

df_stats = (
  df_raw.select(
    stddev(   col('depth') ).alias('depth_stddev'),
    mean( col('depth') ).alias('depth_mean'),
    
    stddev(   col('x') ).alias('x_stddev'),
    mean( col('x') ).alias('x_mean'),
  )
)

# 確認
display(df_stats)

# COMMAND ----------

# DBTITLE 1,depthとxについて、3sigma(3*stddev)を超える値を持つレコードを除外する
# それぞれのstddevを変数に入れておく

_depth_stddev = df_stats.collect()[0]['depth_stddev']
_depth_mean = df_stats.collect()[0]['depth_mean']

_x_stddev = df_stats.collect()[0]['x_stddev']
_x_mean = df_stats.collect()[0]['x_mean']

df_cleaned = (
  df_raw
  .filter( col('depth') >  _depth_mean - 3 * _depth_stddev )
  .filter( col('depth') <  _depth_mean + 3 * _depth_stddev )
  .filter( col('x') >  _x_mean - 3 * _x_stddev )
  .filter( col('x') <  _x_mean + 3 * _x_stddev )
)

# 確認
display(df_cleaned)

# レコード数の確認
print( 'original=>', df_raw.count())
print( 'cleaned =>', df_cleaned.count() )

# COMMAND ----------

# DBTITLE 1,priceごとにS, M, Lのカテゴリをする(カラム追加)
# price <  500  : 'S'
# price <  2000 : 'M'
# price >= 2000 : 'L'

from pyspark.sql.functions import udf

@udf('string')
def price_category(price):
  if price < 500:
    return 'S'
  elif price < 2000:
    return 'M'
  else:
    return 'L'


df_price_categolized = (
  df_cleaned
  .withColumn('price_cat', price_category('price'))
)


display(df_price_categolized)

# COMMAND ----------

# DBTITLE 1,外部テーブルを用意する(結合処理サンプルのため)
# CSVからSparkDataFrameに読み込んでも良いですが、サンプルのため、スクラッチでSpark DataFrameを作成します。

color_desc = []
color_desc.append( {'color_mark': 'A', 'color_desc': 'アップル'} )
color_desc.append( {'color_mark': 'B', 'color_desc': 'ベア'} )
color_desc.append( {'color_mark': 'C', 'color_desc': 'キャット'} )
color_desc.append( {'color_mark': 'D', 'color_desc': 'ドッグ'} )
color_desc.append( {'color_mark': 'E', 'color_desc': 'エレファント'} )
color_desc.append( {'color_mark': 'F', 'color_desc': 'フィッシュ'} )
color_desc.append( {'color_mark': 'G', 'color_desc': 'ゲート'} )
color_desc.append( {'color_mark': 'H', 'color_desc': 'ハット'} )
color_desc.append( {'color_mark': 'I', 'color_desc': 'インク'} )
color_desc.append( {'color_mark': 'J', 'color_desc': 'ジェット'} )
color_desc.append( {'color_mark': 'K', 'color_desc': 'カイト'} )

df_color_desc = spark.createDataFrame(color_desc)
display( df_color_desc )

# COMMAND ----------

# DBTITLE 1,2つのDataframeを結合させる
# 2つのdataframeで以下のcolorカラムをキーとして結合させる
# - df_price_categolized.color
# - df_color_desc.color_mark

df_joined = (
  df_price_categolized.join(df_color_desc,
                            df_price_categolized.color == df_color_desc.color_mark)
)

# 確認
display( df_joined )

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/

# COMMAND ----------

# DBTITLE 1,結果のDataframeをDelta保存して永続化させる
df_joined.write.format('delta').save('dbfs:/FileStore/csv_cook_delta/df_joined.delta')

# COMMAND ----------

# DBTITLE 1,上記のDeltaファイルを読み込む(後続の機械学習などで使用)
df_joined_delta = (
  spark
  .read
  .format('delta')
  .load('dbfs:/FileStore/csv_cook_delta/df_joined.delta')
)

# 確認
display(
  df_joined_delta
)

# COMMAND ----------

# DBTITLE 1,カテゴリー条件で抜き出して、pandas dataframeに変換する
# Color:E かつ price_cat:M のレコードのみを抜き出す
df_filtered = (
  df_joined_delta
  .filter( col('color') == 'E' )
  .filter( col('price_cat') == 'M')
)

# pandas dataframeに変換
df_filtered_pandas = df_filtered.toPandas()

# 確認
df_filtered_pandas
