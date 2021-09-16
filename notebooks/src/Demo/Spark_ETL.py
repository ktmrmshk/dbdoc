# Databricks notebook source
# DBTITLE 1,衝突回避のための変数設定(ハンズオン共有環境のため)
# ユニークな文字列を入力してください
# 例: username = 'kitamura_1112'
username = 'YOUR_UNIQUE_NAME'

print( f'{username}' )

# 前回のファイルを削除(for repeatibility)
dbutils.fs.rm(f'/home/ktmr-handson/{username}/', True)

# COMMAND ----------

# MAGIC %md # 1. データの読み込み

# COMMAND ----------

# MAGIC %md ## CSVファイル

# COMMAND ----------

# DBTITLE 1,ストレージ内ファイルのリスト(実体はS3)
# MAGIC %fs ls /databricks-datasets/nyctaxi/tripdata/yellow/

# COMMAND ----------

df_csv = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .option('inferSchema', True) # schemaは推定
  .load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-*.csv.gz')
  #.load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-10.csv.gz')
)

# 読み込んだデータフレームを確認
display( df_csv )

# COMMAND ----------

# schemaを明示的に指定して実行も可能
schema_csv='''
  VendorID int,
  tpep_pickup_datetime timestamp,
  tpep_dropoff_datetime timestamp,
  passenger_count int,
  trip_distance double,
  RatecodeID int,
  store_and_fwd_flag string,
  PULocationID int,
  DOLocationID int,
  payment_type int,
  fare_amount double,
  extra double,
  mta_tax double,
  tip_amount double,
  tolls_amount double,
  improvement_surcharge double,
  total_amount double,
  congestion_surcharge double
'''

df_csv = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  #.option('inferSchema', True)
  .schema(schema_csv)
  .load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-*.csv.gz')
  #.load('/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-10.csv.gz')
)

# 読み込んだデータフレームを確認
display( df_csv )

# COMMAND ----------

# レコード数
print( f'レコード数 => {df_csv.count()}' )

# COMMAND ----------

df_csv.printSchema()

# COMMAND ----------

# MAGIC %md ## JSONファイル

# COMMAND ----------

df_json = (
  spark
  .read
  .format('json')
  .option('inferSchema', True)
  .load('/mnt/training/ecommerce/events/events-500k.json')
)

# データフレームの確認
display(df_json)

# schemaの確認
df_json2.printSchema()

# COMMAND ----------

# schemaを明示的にしていることも可能

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, LongType, StringType, StructType, StructField

userDefinedSchema = StructType([
  StructField("device", StringType(), True),
  StructField("ecommerce", StructType([
    StructField("purchaseRevenue", DoubleType(), True),
    StructField("total_item_quantity", LongType(), True),
    StructField("unique_items", LongType(), True)
  ]), True),
  StructField("event_name", StringType(), True),
  StructField("event_previous_timestamp", LongType(), True),
  StructField("event_timestamp", LongType(), True),
  StructField("geo", StructType([
    StructField("city", StringType(), True),
    StructField("state", StringType(), True)
  ]), True),
  StructField("items", ArrayType(
    StructType([
      StructField("coupon", StringType(), True),
      StructField("item_id", StringType(), True),
      StructField("item_name", StringType(), True),
      StructField("item_revenue_in_usd", DoubleType(), True),
      StructField("price_in_usd", DoubleType(), True),
      StructField("quantity", LongType(), True)
    ])
  ), True),
  StructField("traffic_source", StringType(), True),
  StructField("user_first_touch_timestamp", LongType(), True),
  StructField("user_id", StringType(), True)
])

# 読み込み
df_json2 = (
  spark
  .read
  .format('json')
  #.option('inferSchema', True)
  .schema(userDefinedSchema)
  .load('/mnt/training/ecommerce/events/events-500k.json')
)

# データフレームの確認
display( df_json2 )

# schemaの確認
df_json2.printSchema()

# COMMAND ----------

# MAGIC %md ## Parquetファイル

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/samples/lending_club/parquet/

# COMMAND ----------

df_parquet = (
  spark
  .read
  .format('parquet')
  .load('/databricks-datasets/samples/lending_club/parquet/')
)

# データフレームの確認
display(df_parquet)

# レコード数
print( f'レコード数 => { df_parquet.count() }' )

# schemaの確認
df_parquet.printSchema()

# COMMAND ----------

# MAGIC %md # 2. データの書き出し

# COMMAND ----------

# MAGIC %md ## Parquet

# COMMAND ----------

# 先ほど読み込んだCSVファイルをparquetで書き出す
(
  df_csv
  .write
  .format('parquet')
  .mode('overwrite')
  #.partitionBy('VendorID')
  .save(f'/home/ktmr-handson/{username}/example.parquet')
)

# COMMAND ----------

#書き込まれたファイルを確認
display(
  dbutils.fs.ls(f'/home/ktmr-handson/{username}/example.parquet')
)

# COMMAND ----------

# MAGIC %md ## Delta Lake
# MAGIC 
# MAGIC Delta形式で保存すると、従来のデータレイクフォーマット(Parquet, Avro, CSVなど)とは異なり、以下の機能性が利用できます。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <div style="float:left; padding-right:60px; margin-top:20px; margin-bottom:200px;">
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta-lake-square-black.jpg" width="220">
# MAGIC </div>
# MAGIC 
# MAGIC <div style="float:left; margin-top:0px; padding:0;">
# MAGIC   <h3>信頼性</h3>
# MAGIC   <br>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC     <li>次世代データフォーマット技術</li>
# MAGIC     <li>トランザクションログによるACIDコンプライアンス</li>
# MAGIC     <li>DMLサポート（更新、削除、マージ）</li>
# MAGIC     <li>データ品質管理　(スキーマージ・エンフォース)</li>
# MAGIC     <li>バッチ処理とストリーム処理の統合</li>
# MAGIC     <li>タイムトラベル (データのバージョン管理)</li>
# MAGIC    </ul>
# MAGIC 
# MAGIC   <h3>パフォーマンス</h3>
# MAGIC   <br>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC      <li>スケーラブルなメタデータ</li>
# MAGIC     <li>コンパクション (Bin-Packing)</li>
# MAGIC     <li>データ・インデックシング</li>
# MAGIC     <li>データ・スキッピング</li>
# MAGIC     <li>ZOrderクラスタリング</li>
# MAGIC     <li>ストリーム処理による低いレイテンシー</li>
# MAGIC   </ul>
# MAGIC </div>

# COMMAND ----------

# 先ほど読み込んだCSVファイルを **Delta形式** で書き出す
(
  df_csv
  .write
  #.format('parquet')
  .format('delta')
  .mode('overwrite')
  #.partitionBy('VendorID')
  .save(f'/home/ktmr-handson/{username}/example.delta')
)

# COMMAND ----------

#書き込まれたファイルを確認
display(
  dbutils.fs.ls(f'/home/ktmr-handson/{username}/example.delta')
)

# COMMAND ----------

# MAGIC %md # 3. ETL処理/データ加工処理

# COMMAND ----------

# DBTITLE 1,基本処理
from pyspark.sql.functions import col, expr, length

df_cleansed = (
  
  df_parquet.select('loan_amnt',  # カラムを選択
            'term',
            'int_rate',
            'grade',
            'addr_state',
            'emp_title',
            'home_ownership',
            'annual_inc',
            'loan_status')
  .withColumn('int_rate', expr('cast(replace(int_rate,"%","") as float)')) # Cast/変換
  .withColumnRenamed('addr_state', 'state') # カラム名変更
  .filter( 'int_rate > 7.5' ) # フィルター
  .filter( length(col('state')) == 2)
  .dropDuplicates()  # 重複レコード削除
)


# 確認
display( df_cleansed )


# COMMAND ----------

# DBTITLE 1,Aggregation
from pyspark.sql.functions import sum, avg, max, min, count, desc

# 州ごと、home_ownershipごとにlaod_amntを集計する
df_agged = (
  df_cleansed
  .groupBy('state', 'home_ownership')
  .agg(
    sum('loan_amnt'),
    avg('loan_amnt'),
    max('loan_amnt'),
    count('*').alias('cnt')
  )
)

# 結果の確認
display( df_agged )

# COMMAND ----------

# DBTITLE 1,(補足) 上記と同じ処理がSQLでも書けます
# SQLから参照するためのtemp viewをDataframeに紐付ける
df_cleansed.createOrReplaceTempView('table_df_cleansed')

# クエリの実行
df_agged_by_sql = spark.sql('''
  SELECT 
    state, 
    home_ownership, 
    sum(loan_amnt), 
    avg(loan_amnt), 
    max(loan_amnt),
    count(*) as cnt
  FROM table_df_cleansed
  GROUP BY state, home_ownership
''')

# 結果の確認
display( df_agged_by_sql )

# COMMAND ----------

# MAGIC %md # 4. データ確認・可視化・分析
# MAGIC 
# MAGIC Databricksのnotebookはインタラクティブにデータの確認、可視化が可能です。

# COMMAND ----------

# テーブル表示
display( df_cleansed )

# COMMAND ----------

# ローン残高のヒストグラム
display( df_cleansed )

# COMMAND ----------

# 州ごとのローン残高の地図を表示
display( df_cleansed )

# COMMAND ----------

# ローン残高、金利、収入の相関関係
display( df_cleansed )

# COMMAND ----------

# MAGIC %md # 5. ユーザー定義関数 (UDF)
# MAGIC 
# MAGIC ## Pandas UDF(Vectorized UDF)
# MAGIC 
# MAGIC Apache Arrowを使用して従来のUDFに比べて高速に実行可能。詳しくは[こちらのドキュメント](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/spark-sql/udf-python-pandas)を参照ください。

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType

# 次の処理をUDFとして定義し、Sparkのデータフレームに適用する
#  入力: col_1, col_2 (数値型)
#  出力: sha256( col_1 * col_2)

# UDFを定義する
def multiply_hash_func(col_1: pd.Series, col_2: pd.Series) -> pd.Series:
  import hashlib
  m = col_1 * col_2 # 掛け合わせる
  hashed = m.apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()) # hash化する
  
  return hashed


# UDFとして登録
multiply_hash_udf = pandas_udf(multiply_hash_func, returnType=StringType())


# Spark DFに適用する( 売り上げ = Quantity * UnitPrice を計算する)
df_applied = (
  df_cleansed
  .selectExpr('*')
  .withColumn('hashed_val', multiply_hash_udf( col('loan_amnt'), col('int_rate')  )) # <= UDFの適用
)

# データフレームの確認
display(df_applied)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (補足)上記のUDFはSQL上でも使用可能です

# COMMAND ----------

# SQLで使うために登録する
spark.udf.register('multiply_hash_udf_for_sql', multiply_hash_udf)

# SQLから参照するためのtemp viewをDataframeに紐付ける
df_cleansed.createOrReplaceTempView('table_df_cleansed')

# SQLで実行
df_applied_by_sql = spark.sql('''
  SELECT
    *,
    multiply_hash_udf_for_sql( loan_amnt,  int_rate ) as hashed_val
  FROM
    table_df_cleansed
''')


# 結果の確認
display( df_applied_by_sql )


# COMMAND ----------

# MAGIC %md # 6. Jobの登録
# MAGIC 
# MAGIC * このNotebookをJobに登録することで、定期的にNotebook上の処理を実行することができます。
# MAGIC * API経由でJobを外部からkick(実行開始)することが可能です。

# COMMAND ----------

# MAGIC %md # 7. DatabricskのUI機能
# MAGIC * SparkI UI
# MAGIC * コラボレーション機能・Notebookの共有
# MAGIC * Revision管理、Git連携
# MAGIC * 外部SparkコードをDatabircks上で実行(Databricks Connect)
# MAGIC * クラスタ管理・モニタ
# MAGIC * クラスタのオートスケーリング

# COMMAND ----------


