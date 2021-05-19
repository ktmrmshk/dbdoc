# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、協調フィルタリングによるレコメンダーを実施するために使用するデータセットを準備することです。 このノートブックは **Databricks 7.1+ クラスタ** で実行する必要があります。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # イントロダクション
# MAGIC 
# MAGIC コラボレーティブ・フィルターは、現代のレコメンデーション体験を実現する重要な要素です。「 ***あなたに似たお客様はこんなものも買っています***」というタイプのレコメンデーションは、関連性の高いお客様の購買パターンに基づいて、興味を引きそうな商品を特定する重要な手段となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
from pyspark.sql.types import *
from pyspark.sql.functions import count, countDistinct, avg, log, lit, expr

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: Load the Data
# MAGIC 
# MAGIC The basic building block of this kind of recommendation is customer transaction data. To provide us data of this type, we'll be using the popular [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis). This dataset provides cart-level details on over 3 million grocery orders placed by over 200,000 Instacart users across of portfolio of nearly 50,000 products.
# MAGIC 
# MAGIC **NOTE** Due to the terms and conditions by which these data are made available, anyone interested in recreating this work will need to download the data files from Kaggle and upload them to a folder structure as described below.
# MAGIC 
# MAGIC The primary data files available for download are organized as follows under a pre-defined [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) that we have named */mnt/instacart*:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC Read into dataframes, these files form the following data model which captures the products customers have included in individual transactions:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC 
# MAGIC We will apply minimal transformations to this data, persisting it to the Delta Lake format for speedier access:

# COMMAND ----------

# DBTITLE 1,Create Database
_ = spark.sql('CREATE DATABASE IF NOT EXISTS instacart')

# COMMAND ----------

# MAGIC %md #IMPORTANT
# MAGIC 
# MAGIC **NOTE** The orders data set is pre-split into *prior* and *training* datasets.  Because date information in this dataset is very limited, we'll need to work with these pre-defined splits.  We'll treat the *prior* dataset as our ***calibration*** dataset and we'll treat the *training* dataset as our ***evaluation*** dataset. To minimize confusion, we'll rename these as part of our data preparation steps.

# COMMAND ----------

# DBTITLE 1,Orders
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.orders')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/orders', ignore_errors=True)

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

# rename eval_set entries
orders_transformed = (
  orders
    .withColumn('split', expr("CASE eval_set WHEN 'prior' THEN 'calibration' WHEN 'train' THEN 'evaluation' ELSE NULL END"))
    .drop('eval_set')
  )

# write data to delta
(
  orders_transformed
    .write
    .format('delta')
    .mode('overwrite')
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
  spark.table('instacart.orders')
  )

# COMMAND ----------

# DBTITLE 1,Products
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.products')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/products', ignore_errors=True)

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

# DBTITLE 1,Order Products
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.order_products')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/order_products', ignore_errors=True)

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

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/departments', ignore_errors=True)

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

# DBTITLE 1,Aisles
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS instacart.aisles')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/silver/aisles', ignore_errors=True)

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

# MAGIC %md # Step 2: Derive Product *Ratings*
# MAGIC 
# MAGIC For our collaborative filter (CF), we need a way to understand user preferences for individual products. In some scenarios, explicit user ratings, such as a 3 out of 5 stars rating, may be provided, but not every interaction receives a rating and in many transactional engagements the idea of asking customers for such ratings just seems out of place. In these scenarios, we might use other user-generated data to indicate product preferences. In the context of the Instacart dataset, the frequency of product purchases by a user may serve as such an indicator:

# COMMAND ----------

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/gold/ratings__user_product_orders', ignore_errors=True)

# identify number of times product purchased by user
user_product_orders = (
  spark
    .table('instacart.orders')
    .join(spark.table('instacart.order_products'), on='order_id')
    .groupBy('user_id', 'product_id', 'split')
    .agg( count(lit(1)).alias('purchases') )
  )

# write data to delta
(
  user_product_orders
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/ratings__user_product_orders')
  )

# display results
display(
  spark.sql('''
    SELECT * 
    FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders` 
    ORDER BY split, user_id, product_id
    ''')
)

# COMMAND ----------

# MAGIC %md Using product purchases as *implied ratings* presents us with a scaling problem.  Consider a scenario where a user purchases a given product 10 times while another user purchases a product 20 times.  Does the first user have a stronger preference for the product?  What if we new the first customer has made 10 purchases in total so that this product was included in each checkout event while the second user had made 50 total purchases, only 20 of which included the product of interest?  Does our understanding of the users preferences change in light of this additional information?
# MAGIC 
# MAGIC Rescaling our data to account for differences in overall purchase frequency will provide us a more reliable basis for the comparison of users. There are several options for doing this, but because of how we intend to measure the similarity between users (to provide the basis of collaborative filtering), our preference will be to use what is referred to as L2-normalization.
# MAGIC 
# MAGIC To understand L2-normalization, consider two users who have purchased products X and Y. The first user has purchased product X 10 times and product Y 5 times. The second user has purchased products X and Y 20 times each.  We might plot these purchases (with product X on the x-axis and product Y on the y-axis) as follows:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm01.png' width=380>
# MAGIC 
# MAGIC To determine similarity, we'll be measuring the (Euclidean) distance between the points formed at the intersection of these two axes, *i.e.* the peak of the two triangles in the graphic.  Without rescaling, the first user resides about 11 units from the origin and the second user resides about 28 units.  Calculating the distance between these two users in this space would provide a measure of both differing product preferences AND purchase frequencies. Rescaling the distance each user resides from the origin of the space eliminates the differences related to purchase frequencies, allowing us to focus on differences in product preferences:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm02.png' width=400>
# MAGIC 
# MAGIC The rescaling is achieved by calculating the Euclidean distance between each user and the origin - there's no need to limit ourselves to two-dimensions for this math to work - and then dividing each product-specific value for that user by this distance which is referred to as the L2-norm.  Here, we apply the L2-norm to our implied ratings:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS instacart.user_ratings;
# MAGIC 
# MAGIC CREATE VIEW instacart.user_ratings 
# MAGIC AS
# MAGIC   WITH ratings AS (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       user_id,
# MAGIC       product_id,
# MAGIC       SUM(purchases) as purchases
# MAGIC     FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders`
# MAGIC     GROUP BY split, user_id, product_id
# MAGIC     )
# MAGIC   SELECT
# MAGIC     a.split,
# MAGIC     a.user_id,
# MAGIC     a.product_id,
# MAGIC     a.purchases,
# MAGIC     a.purchases/b.l2_norm as normalized_purchases
# MAGIC   FROM ratings a
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       user_id,
# MAGIC       POW( 
# MAGIC         SUM(POW(purchases,2)),
# MAGIC         0.5
# MAGIC         ) as l2_norm
# MAGIC     FROM ratings
# MAGIC     GROUP BY user_id, split
# MAGIC     ) b
# MAGIC     ON a.user_id=b.user_id AND a.split=b.split;
# MAGIC   
# MAGIC SELECT * FROM instacart.user_ratings ORDER BY user_id, split, product_id;

# COMMAND ----------

# MAGIC %md You may have noted that we elected to implement these calculations through a view.  If we consider the values for a user must be recalculated with each purchase event by that user as that event will impact the value of the L2-norm by which each implied rating is adjusted. Persisting raw purchase counts in our base *ratings* table provides us an easy way to incrementally add new information to this table without having to re-traverse a user's entire purchase history.  Aggregating and normalizing the values in that table on the fly through a view gives us an easy way to extract normalized data with less ETL effort.
# MAGIC 
# MAGIC It's important to consider which data is included in these calculations. Depending on your scenario, it might be appropriate to limit the transaction history from which these *implied ratings* are derived to a period within which expressed preferences would be consistent with the user's preferences in the period over which the recommender might be used.  In some scenarios, this may mean limiting historical data to a month, quarter, year, etc.  In other scenarios, this may mean limiting historical data to periods with comparable seasonal components as the current or impending period.  For example, a user may have a strong preference for pumpkin spice flavored products in the Fall but may not be really keen on it during the Summer months.  For demonstration purposes, we'll just use the whole transaction history as the basis of our ratings but this is a point you'd want to carefully consider for a real-world implementation.

# COMMAND ----------

# MAGIC %md # Step 3: Derive Naive Product *Ratings*
# MAGIC 
# MAGIC A common practice when evaluating a recommender is to compare it to a prior or alternative recommendation engine to see which better helps the organization achieve its goals. To provide us a starting point for such comparisons, we might consider using overall product popularity as the basis for making *naive* collaborative recommendations. Here, we calculate normalized product ratings based on overall purchase frequencies to enable this work:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS instacart.naive_ratings;
# MAGIC 
# MAGIC CREATE VIEW instacart.naive_ratings 
# MAGIC AS
# MAGIC   WITH ratings AS (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       product_id,
# MAGIC       SUM(purchases) as purchases
# MAGIC     FROM DELTA.`/mnt/instacart/gold/ratings__user_product_orders`
# MAGIC     GROUP BY split, product_id
# MAGIC     )
# MAGIC   SELECT
# MAGIC     a.split,
# MAGIC     a.product_id,
# MAGIC     a.purchases,
# MAGIC     a.purchases/b.l2_norm as normalized_purchases
# MAGIC   FROM ratings a
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       POW( 
# MAGIC         SUM(POW(purchases,2)),
# MAGIC         0.5
# MAGIC         ) as l2_norm
# MAGIC     FROM ratings
# MAGIC     GROUP BY split
# MAGIC     ) b
# MAGIC     ON a.split=b.split;
# MAGIC   
# MAGIC SELECT * FROM instacart.naive_ratings ORDER BY split, product_id;
