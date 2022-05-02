# Databricks notebook source
# DBTITLE 1,Notes for Presenter
# MAGIC %md このデモは、技術的な詳細というより、ビジネス上の成果に関心があるビジネスアラインドのペルソナを対象としています。 そのため、このノートブックのコードの多くは隠され、ロジック自体も合理化されており、彼らが関心を持つポイントにすぐに到達できるようになっています。 より技術的な説明を必要とされる方は、[the solution accelerator](https://databricks.com/blog/2021/04/06/fine-grained-time-series-forecasting-at-scale-with-facebook-prophet-and-apache-spark-updated-for-spark-3.html)に関連するノートブックの利用をご検討ください。
# MAGIC 
# MAGIC このノートブックは、**Databricks 7.3 LTS**ランタイムで動作するように開発されています。 実行する前に、必ず[このデータセット](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)をダウンロード、解凍し、Databricksにアップロードしてください。 データセットはCSVで */FileStore/demand_forecast/train* に保存してください。
# MAGIC 
# MAGIC 最後に、*View.Results Only*を必ず選択する。最後に、このノートブックを顧客に提示する前に、*View: Results Only* を選択してください。

# COMMAND ----------

# DBTITLE 0,Install Required Libraries
# MAGIC %pip install pystan==2.19.1.1  # per https://github.com/facebook/prophet/commit/82f3399409b7646c49280688f59e5a3d2c936d39#comments
# MAGIC %pip install fbprophet==0.6

# COMMAND ----------

from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.window import Window

from fbprophet import Prophet
import logging

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# COMMAND ----------

# MAGIC %md ## Step 1: データを検証する
# MAGIC 
# MAGIC 予測を作成するデータセットは、5年間にわたる10店舗の50商品の日次売上データから構成されています。

# COMMAND ----------

# DBTITLE 1,Review Raw Data
# structure of the training data set
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# read the training file into a dataframe
train = spark.read.csv(
  'dbfs:/FileStore/demand_forecast/train/train.csv', 
  header=True, 
  schema=train_schema
  )

# make the dataframe queriable as a temporary view
train.createOrReplaceTempView('train')

# show data
display(train)

# COMMAND ----------

# MAGIC %md 予測を行う際の典型的な例として、年単位と週単位の両方で、データにトレンドと季節性があるかどうかを調べたいと思います。

# COMMAND ----------

# DBTITLE 1,View Yearly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# DBTITLE 1,View Monthly Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# DBTITLE 1,View Weekday Trends
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %md ## Step 2: 需要予測を実施する
# MAGIC 
# MAGIC 時系列予測を構築するための強力なパターンがあるように見えます。 しかし、10店舗と50アイテムでは、500店舗とアイテムの予測を行うために500のモデルをトレーニングする必要があります。

# COMMAND ----------

# DBTITLE 1,Get DataSet Metrics
# MAGIC %sql -- get dataset metrics
# MAGIC 
# MAGIC SELECT 
# MAGIC   COUNT(DISTINCT store) as stores,
# MAGIC   COUNT(DISTINCT item) as items,
# MAGIC   COUNT(DISTINCT year(date)) as years,
# MAGIC   COUNT(*) as records
# MAGIC FROM train;

# COMMAND ----------

# MAGIC %md 従来のアプローチは、データを集計し、集計されたデータセットから予測を生成するものでした。 もし10店舗すべてのデータを集約すれば、必要なモデル数は50に減り、これは歴史的に我々のビジネスにとってより現実的なものでした。 これを店舗レベルに戻すには、販売台数に基づく割り当てを使用します。

# COMMAND ----------

# DBTITLE 1,Generate Allocated Forecasts (Aggregated Stores)
# allocation ratios
ratios = (
  spark.sql('''
    SELECT
      store,
      item,
      sales / SUM(sales) OVER(PARTITION BY item) as ratio
    FROM (
      SELECT 
        store,
        item,
        SUM(sales) as sales
      FROM train
      GROUP BY
        store, item
        ) 
    ''')
    )

# define forecasting function
result_schema =StructType([
  StructField('ds',DateType()),
  StructField('item',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

def forecast_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # train the model
  model.fit( history_pd )
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # get relevant fields from history
  h_pd = history_pd[['ds','item','y']].set_index('ds')
  
  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[ ['ds', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

forecast = (
  spark
    .table('train')
    .withColumnRenamed('date','ds')
    .groupBy('item','ds')
      .agg(f.sum('sales').alias('y'))
    .orderBy('item','ds')
    .repartition('item')
    .groupBy('item')
      .applyInPandas(forecast_item, schema=result_schema)
    .withColumn('training_date', f.current_date())
  )

results = (
  forecast
    .join(ratios, on='item')
    .withColumn('yhat',f.expr('yhat * ratio'))
    .selectExpr('ds as date','store','item','y as sales', 'yhat as forecast', 'training_date') 
    )

_ = (
  results
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('allocated_forecasts')
  )

display(spark.table('allocated_forecasts'))

# COMMAND ----------

# MAGIC %md Databricksでは、クラウドを活用して、実際に必要な500個のモデルを提供する可能です!

# COMMAND ----------

# DBTITLE 1,Generate Fine-Grained Forecasts
# retrieve historical data
sql_statement = '''
  SELECT
    store,
    item,
    CAST(date as date) as ds,
    SUM(sales) as y
  FROM train
  GROUP BY store, item, ds
  ORDER BY store, item, ds
  '''

store_item_history = (
  spark
    .sql( sql_statement )
    .repartition(sc.defaultParallelism, ['store', 'item'])
  ).cache()

# define forecasting function
result_schema =StructType([
  StructField('ds',DateType()),
  StructField('store',IntegerType()),
  StructField('item',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # train the model
  model.fit( history_pd )
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # get relevant fields from history
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# generate forecast
results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema=result_schema)
    .withColumn('training_date', f.current_date() )
    .withColumnRenamed('ds','date')
    .withColumnRenamed('y','sales')
    .withColumnRenamed('yhat','forecast')
    .withColumnRenamed('yhat_upper','forecast_upper')
    .withColumnRenamed('yhat_lower','forecast_lower')
    )

_ = (
  results
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('forecasts')
  )

display(spark.table('forecasts').drop('forecast_upper','forecast_lower'))

# COMMAND ----------

# MAGIC %md このDatabricksを使ったアプローチは、問題に割り当てるリソースの数を決めることができ、それがそのままオペレーションを完了するための時間になるという点で有効です。 ここでは、500のモデルを実行したときの処理時間について、環境の大きさの違いを見ていますが、当社の最大の顧客の多くは、これと同じパターンを使って毎日1～2時間以内に数百万件の予測を完了していることに留意してください。

# COMMAND ----------

# DBTITLE 1,Examine Scalability
# Test runs on Azure F4s_v2 - 4 cores, 8 GB RAM
tests_pd = pd.DataFrame(
    [ (1, 20.77 * 60),
      (2, 11.14 * 60),
      (3, 7.46 * 60),
      (4, 6.11 * 60),
      (5, 5.01 * 60),
      (6, 4.90 * 60),
      (8, 3.81 * 60),
      (10, 3.20 * 60),
      (12, 2.80 * 60),
      (15, 2.26 * 60) ],
    columns = ['workers', 'seconds']
    )

tests_pd['cores'] = tests_pd['workers'] * 4 # 4-cores per worker VM

display(tests_pd)

# COMMAND ----------

# MAGIC %md それぞれの結果を比較すると、きめ細かい予測は、私たちが捕らえたいと思っていた局所的な変動をもたらすのに対し、配分法は単に店舗間予測のスケーリングされた変動を返すだけであることがわかります。 これらの変動は、地域の需要の違いを表しており、利益を最大化するためには、オペレーションを微調整する必要があります。

# COMMAND ----------

# DBTITLE 1,Visualize Allocated Forecasts for Item 1 at Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   store,
# MAGIC   date,
# MAGIC   forecast
# MAGIC FROM allocated_forecasts
# MAGIC WHERE item = 1 AND 
# MAGIC       date >= '2018-01-01' AND 
# MAGIC       training_date=current_date()
# MAGIC ORDER BY date, store

# COMMAND ----------

# DBTITLE 1,Visualize Fine Grained Forecasts for Item 1 at Each Store
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   forecast
# MAGIC FROM forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY date, store

# COMMAND ----------

# MAGIC %md ## Step 3: アナリストへのデータ提供/共有
# MAGIC 
# MAGIC Databricksはこれらの予測を作成するのに必要な時間を短縮するのに優れていますが、アナリストはどのようにそれを利用するでしょうか。このノートブックにあるネイティブな可視化機能はすでにご覧の通りです。 これは、データサイエンティストの作業を支援するための機能です。
# MAGIC 
# MAGIC アナリストの場合は、[Databricks' SQL Dashboard](https://adb-2704554918254528.8.azuredatabricks.net/sql/dashboards/fafa7c3f-35e0-4b4c-925e-04e305155678?o=2704554918254528)を活用して、結果を提示することもできます。

# COMMAND ----------

# MAGIC %md <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_dashboard.PNG' width=800>

# COMMAND ----------

# MAGIC %md また、TableauやPower BIなどのツールでデータを表示したいと思うこともあります。

# COMMAND ----------

# MAGIC %md <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_powerbi.PNG' width=800>

# COMMAND ----------

# MAGIC %md これらのデータをExcelで提示することも可能です。

# COMMAND ----------

# MAGIC %md 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/forecasting_excel.PNG' width=800>
