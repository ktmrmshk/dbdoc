# Databricks notebook source
# MAGIC %md このノートブックの目的は、ディープ＆ワイド協調フィルタレコメンダーを構築するために使用するデータセットを準備することです。 このノートブックは **Databricks 8.1+ ML cluster** で実行する必要があります。

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC 協調フィルタは、ユーザー間の類似性を利用して推薦を行うものです。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">
# MAGIC 
# MAGIC モデルベースの協調フィルタは，類似するユーザ間の製品評価（明示的または黙示的）を加重平均するメモリベースの協調フィルタとは異なり，ユーザと製品の組み合わせに関連する特徴を利用して，特定のユーザが特定のアイテムをクリックまたは購入することを予測します． このようなモデルを構築するためには、ユーザーとそのユーザーが購入した商品に関する情報が必要となります。

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql import window as w 

# COMMAND ----------

# MAGIC %md # Step 1: データのロード
# MAGIC 
# MAGIC 協調フィルタの基本的な構成要素は、顧客の識別子を含むトランザクションデータです。人気の高い[Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis)は、約2年間に渡って約50,000の製品のポートフォリオに対して200,000人以上のInstacartユーザーが行った300万件以上の食料品の注文を含む、このようなデータの素晴らしいコレクションを提供しています。これは、[既刊のノートブックセット](https://databricks.com/blog/2020/12/18/personalizing-the-customer-experience-with-recommendations.html)に記載されている、記憶ベースの協調フィルタの構築に使用されたデータセットと同じものであり、ここで検討されている技術との良い比較になります。
# MAGIC 
# MAGIC **注**データの提供条件により、この作品を再現するには、Kaggleからデータファイルをダウンロードして、以下のようなフォルダ構造にアップロードする必要があります。
# MAGIC 
# MAGIC ダウンロード可能な主要データファイルは、*/mnt/instacart*と名付けた事前定義の[マウントポイント](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)の下に以下のように整理されています。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これらのファイルはデータフレームに読み込まれ、お客様が個々のトランザクションに含めた商品をキャプチャする以下のデータモデルを形成します。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC 
# MAGIC このデータには最小限の変換を行い、高速なアクセスのためにDelta Lakeフォーマットに変換します。

# COMMAND ----------

# DBTITLE 1,データベースの作成
_ = spark.sql('CREATE DATABASE IF NOT EXISTS instacart')

# COMMAND ----------

# MAGIC %md 注文データは、「先行」と「トレーニング」の評価セットに分けられます。「トレーニング」データセットは、ある顧客に関連する注文の全体的な流れの中で、最後に発注された注文を表します。 先行」データセットは、「トレーニング」よりも前に発注された注文を表します。 このデータを使った以前のノートブックでは、データの使用方法に合わせて用語を整理するために、*prior*と*training*の評価セットをそれぞれ*calibration*と*evaluation*と再表示していました。 ここでは、現在のモデル化のニーズに合わせて、*prior*と*training*の名称を維持します。
# MAGIC 
# MAGIC このデータセットには、「*days_prior_to_last_order*」というフィールドを追加します。これは、ある注文から「*training*」インスタンスを表す注文までの日数を計算するものです。このフィールドは、最終注文の前に異なる間隔で行われる購入に関する機能を開発する際に役立ちます。 その他のテーブルは、スキーマを変更することなくデータベースに導入されますが、後でクエリのパフォーマンスを向上させるために、基本的なフォーマットをCSVからdelta lakeに変換するだけです。

# COMMAND ----------

# DBTITLE 1,注文
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.orders')

# define schema for incoming data
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

# read data from csv
orders = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/orders',
      header=True,
      schema=orders_schema
      )
  )

# calculate days until final purchase 
win = (
  w.Window.partitionBy('user_id').orderBy(f.col('order_number').desc())
  )

orders_enhanced = (
    orders
      .withColumn(
        'days_prior_to_last_order', 
        f.sum('days_since_prior_order').over(win) - f.coalesce(f.col('days_since_prior_order'),f.lit(0))
        ) 
  )

# write data to delta
(
  orders_enhanced
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/mnt/instacart/silver/orders')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.orders
  USING DELTA
  LOCATION '/mnt/instacart/silver/orders'
  ''')

# present the data for review
display(
  spark
    .table('instacart.orders')
    .orderBy('user_id','order_number')
  )

# COMMAND ----------

# DBTITLE 1,製品
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.products')

# define schema for incoming data
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

# read data from csv
products = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/products',
      header=True,
      schema=products_schema
      )
  )

# write data to delta
(
  products
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/mnt/instacart/silver/products')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.products
  USING DELTA
  LOCATION '/mnt/instacart/silver/products'
  ''')

# present the data for review
display(
  spark.table('instacart.products')
  )

# COMMAND ----------

# DBTITLE 1,製品の注文
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.order_products')

# define schema for incoming data
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

# read data from csv
order_products = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/order_products',
      header=True,
      schema=order_products_schema
      )
  )

# write data to delta
(
  order_products
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/mnt/instacart/silver/order_products')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.order_products
  USING DELTA
  LOCATION '/mnt/instacart/silver/order_products'
  ''')

# present the data for review
display(
  spark.table('instacart.order_products')
  )

# COMMAND ----------

# DBTITLE 1,Departments
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.departments')

# define schema for incoming data
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

# read data from csv
departments = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/departments',
      header=True,
      schema=departments_schema
      )
  )

# write data to delta
(
  departments
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/mnt/instacart/silver/departments')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.departments
  USING DELTA
  LOCATION '/mnt/instacart/silver/departments'
  ''')

# present the data for review
display(
  spark.table('instacart.departments')
  )

# COMMAND ----------

# DBTITLE 1,通路
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.aisles')

# define schema for incoming data
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

# read data from csv
aisles = (
  spark
    .read
    .csv(
      '/mnt/instacart/bronze/aisles',
      header=True,
      schema=aisles_schema
      )
  )

# write data to delta
(
  aisles
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/mnt/instacart/silver/aisles')
  )

# make accessible as spark sql table
_ = spark.sql('''
  CREATE TABLE instacart.aisles
  USING DELTA
  LOCATION '/mnt/instacart/silver/aisles'
  ''')

# present the data for review
display(
  spark.table('instacart.aisles')
  )

# COMMAND ----------

# MAGIC %md # Step 2: 注文詳細の結合
# MAGIC 
# MAGIC データが読み込まれたら、ビューを使って注文の詳細をフラットにします。 これにより、機能設計時のデータへのアクセスが非常に容易になります。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS instacart.order_details;
# MAGIC 
# MAGIC CREATE VIEW instacart.order_details as
# MAGIC   SELECT
# MAGIC     a.eval_set,
# MAGIC     a.user_id,
# MAGIC     a.order_number,
# MAGIC     a.order_id,
# MAGIC     a.order_dow,
# MAGIC     a.order_hour_of_day,
# MAGIC     a.days_since_prior_order,
# MAGIC     a.days_prior_to_last_order,
# MAGIC     b.product_id,
# MAGIC     c.aisle_id,
# MAGIC     c.department_id,
# MAGIC     b.reordered
# MAGIC   FROM instacart.orders a
# MAGIC   INNER JOIN instacart.order_products b
# MAGIC     ON a.order_id=b.order_id
# MAGIC   INNER JOIN instacart.products c
# MAGIC     ON b.product_id=c.product_id;
# MAGIC     
# MAGIC SELECT *
# MAGIC FROM instacart.order_details;

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
