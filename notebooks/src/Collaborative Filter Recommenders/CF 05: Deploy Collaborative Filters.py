# Databricks notebook source
# MAGIC %md The purpose of this notebook is to explore how the collaborative-filter recommenders developed in previous notebooks might be operationalized. This notebook should be run on a **Databricks ML 7.1+ cluster**.

# COMMAND ----------

# MAGIC %md # Introduction 
# MAGIC 
# MAGIC The deployment pattern for any recommender is specific to the volumes and volatility of the data on which they are based. The specific business scenarios enabled with the recommenders also affect how you should implement the recommendation platform. In this notebook, we'll explore the mechanics of deploying both user-based and item-based collaborative filters in a manner we believe aligns with some common scenarios but in no way are we suggesting you should deploy a user-based or item-based recommender exactly as demonstrated here.  You are strongly encouraged to discuss the deployment of any recommender system with the developers and architects responsible for the applications that will use the recommendations and, just as importantly, the business stakeholders responsible for the outcomes they are to drive.  With that in mind, let's take a quick look at the deployment pattern explored in this notebook:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_pipeline.png" width="700">
# MAGIC 
# MAGIC With this deployment pattern, we envision a daily, weekly or even monthly process responsible for the generation of user- and/or product-pairs.  These pairs along with ratings data and product data may then be replicated to a relational database (or other data store) where tuned queries are then used to produce the recommendations presented in an application. 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark import keyword_only
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import HasInputCols, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.feature import BucketedRandomProjectionLSH, SQLTransformer
from pyspark.ml.linalg import Vector, Vectors, VectorUDT

import mlflow.spark

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, max, collect_list, lit, expr, coalesce, pow, sum, count
from pyspark.sql.types import *

from delta.tables import *

import pandas as pd
import numpy as np

import math
import shutil

# COMMAND ----------

# MAGIC %md # Step 1a: Assemble User-Pairs
# MAGIC 
# MAGIC Our first step in generating user-based collaborative filter recommendations is to assemble ratings vectors for each user and assign each vector to an LSH bucket.  In many solutions, such steps would be organized as a [pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html) but in the batch scenario we envision here, it may be easier to maintain the code within a script or notebook. (For an example of how pipelines may be employed with LSH, please see the final content-based recommender notebook.)
# MAGIC 
# MAGIC To get us started, we'll need to retrieve details on the users that will be part of our solution. Here we are leveraging the full set of user data from our calibration period.  In a real world implementation, this set may be constrained using data from one or more prior periods that take into consideration seasonal variations in buying patterns and which are relevant to the future period over which we intent to make recommendations:

# COMMAND ----------

# DBTITLE 1,Retrieve User Ratings from Which to Construct Recommendations
user_ratings = spark.sql('''
  SELECT
    user_id,
    product_id,
    normalized_purchases,
    (SELECT max(product_id) + 1 FROM instacart.products) as size
  FROM instacart.user_ratings
  WHERE split = 'calibration'
  ''')

# COMMAND ----------

# DBTITLE 1,Convert Ratings to Feature Vectors
# define and register UDF for vector construction
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    ind, val = zip(*sorted(zip(index_list, value_list)))
    return Vectors.sparse(size, ind, val)
    
_ = spark.udf.register('to_vector', to_vector)

# assemble user vectors
user_features = (
  user_ratings
    .groupBy('user_id')
      .agg(
          collect_list('product_id').alias('index_list'),
          collect_list('normalized_purchases').alias('value_list')
          )
      .crossJoin(
          spark
            .table('instacart.products')
            .groupBy()
              .agg( max('product_id').alias('max_product_id'))
              .withColumn('size', expr('max_product_id+1'))
              .select('size')
          )
      .withColumn('features', expr('to_vector(size, index_list, value_list)'))
      .select('user_id', 'features')
  )

# COMMAND ----------

# DBTITLE 1,Assign LSH Buckets to User Vectors
# configure lsh transform
fitted_lsh = (
  BucketedRandomProjectionLSH(
    inputCol = 'features', 
    outputCol = 'hash', 
    numHashTables = 5, 
    bucketLength = 0.0025
    )
    .fit(user_features)
  )

# assign buckets
user_features_bucketed = fitted_lsh.transform(user_features)

# clean up any older copies of data
shutil.rmtree('/dbfs/mnt/instacart/tmp/user_features', ignore_errors=True)

# persist data for later use
(
  user_features_bucketed
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/tmp/user_features_bucketed')
  )

# COMMAND ----------

# MAGIC %md The last step in the prior cell persists our LSH-bucketed data to disk.  This will provide us a consistent basis for restarting the next set of long-running steps should a problem be encountered during their execution.
# MAGIC 
# MAGIC In our next step, we'll use the *approxSimiliarityJoin()* to return users within a particular distance of a subset of our user population. We'll set the threshold fairly high to ensure we return a large number of users but then limit the users we move forward with the to 10 most similar users (plus the target user him- or herself) to emulate what we might receive from performing a nearest neighbor query.  The reason we are approaching the data retrieval problem in this manner as opposed to using the *approxNearnestNeighbor()* method explored in prior notebooks is that the similarity join is built for the retreival of data for multiple users (as opposed to single user retrieval with the nearest neighbors technique), making this approach significantly faster:

# COMMAND ----------

# DBTITLE 1,Define Function to Limit User-Pairs to Top 10
def get_top_users( data ):
  '''the incoming dataset is expected to have the following structure: user_a, user_b, distance''' 
  
  rows_to_return = 11 # limit to top 10 users (+1 for self)
  min_score = 1 / (1+math.sqrt(2))
  
  data['sim']= 1/(1+data['distance'])
  data['similarity']= (data['sim']-min_score)/(1-min_score)
  ret = data[['user_id', 'paired_user_id', 'similarity']]
  
  return ret.sort_values(by=['similarity'], ascending=False).iloc[0:rows_to_return] # might be faster ways to implement this sort/trunc

# COMMAND ----------

# DBTITLE 1,Initialize Output Table
_ = spark.sql('DROP TABLE IF EXISTS instacart.user_pairs')

shutil.rmtree('/dbfs/mnt/instacart/gold/user_pairs', ignore_errors=True)

_ = spark.sql('''
  CREATE TABLE instacart.user_pairs (
    user_id int,
    paired_user_id int,
    similarity double
    )
  USING delta
  LOCATION '/mnt/instacart/gold/user_pairs'
  ''')

# COMMAND ----------

# DBTITLE 1,Generate User-Pairs
# retrieve hashed users
b = (
  spark
    .table('DELTA.`/mnt/instacart/tmp/user_features_bucketed`')
  ).cache()

max_user_per_cycle = 5000
i = 0

# loop until break is tripped
while 1 > 0:
  
  # enumerate cycles
  i += 1 
  
  # get users to generate pairs for
  a = (
    spark
      .table('DELTA.`/mnt/instacart/tmp/user_features_bucketed`')
      .join( spark.table('instacart.user_pairs'), on='user_id', how='left_anti')  # user_a not in user_pairs table
      .limit(max_user_per_cycle)
    )

  # continue?
  if a.limit(1).count() == 0:
    break
  else:
    print('{0}: {1}'.format(i, i * max_user_per_cycle))

  # generate pairs
  user_pairs = (
    fitted_lsh.approxSimilarityJoin(
      a,
      b,
      threshold = 1.4,
      distCol = 'distance'
      )
      .selectExpr(
        'datasetA.user_id as user_id',
        'datasetB.user_id as paired_user_id',
        'distance'
        )
      .groupBy('user_id')
        .applyInPandas(
          get_top_users, 
          schema='''
            user_id int,
            paired_user_id int,
            similarity double
            ''')
      .write
      .format('delta')
      .mode('append')
      .save('/mnt/instacart/gold/user_pairs')
      )

# print count of pairs
print( 'Total pairs generated: {0}'.format(spark.table('instacart.user_pairs').count()) )

# COMMAND ----------

# DBTITLE 1,Display User-Pairs
display(
  spark.table('instacart.user_pairs')
  )

# COMMAND ----------

# MAGIC %md # Step 1b: Assemble Product-Pairs
# MAGIC 
# MAGIC Our first step in building an item-based collaborative filter is to assemble the product pairs for the prior relevant period.  Unlike the generation of user-pairs, we are not using any bucketing techniques and instead will perform a brute-force generation of pairs actually observed in the data.  To limit the number of calculations performed, we'll perform a non-redundant comparison of product A to product B and then simply insert the inverted product B to product A comparison along with the self-comparison, *i.e.* product A to product A, data into the output table. This pattern was explored in the last notebook.
# MAGIC 
# MAGIC It's important to note that unlike the last notebook, we are implementing a filter to eliminate product pairs associated with fewer than 6 users.  In the last notebook, the filter was implemented during the generation of the recommendations to allow us to explore the ideal setting for our dataset.  Here, we are moving the filter further up into the data processing pipeline to reduce the both data processing and query times:

# COMMAND ----------

# DBTITLE 1,Calculate Product-Pair Similarities
def compare_products( data ):
  '''
  the incoming dataset is expected to have the following structure:
     product_a, product_b, size, values_a, values_b
  '''
  
  def normalize_vector(v):
    norm = Vectors.norm(v, 2)
    ret = v.copy()
    n = v.size
    for i in range(0,n):
      ret[i] /= norm
    return ret
    
  # list to hold results
  results = []
  
  # for each entry in this subset of data ...
  for row in data.itertuples(index=False):
    
    # retrieve data from incoming dataset
    # -----------------------------------------------------------
    product_a = row.product_a
    values_a = row.values_a
    
    product_b = row.product_b
    values_b = row.values_b
    
    size = row.size # this value is not used but is simply passed-through
    # -----------------------------------------------------------
    
    # construct data structures for user comparisons
    # -----------------------------------------------------------
    a = Vectors.dense(values_a)
    a_norm = normalize_vector(a)
    
    b = Vectors.dense(values_b)
    b_norm = normalize_vector(b)
    # -----------------------------------------------------------
    
    # calc distance and similarity
    # -----------------------------------------------------------
    distance = math.sqrt(Vectors.squared_distance(a_norm, b_norm))
    similarity = 1 / (1 + distance)
    similarity_min = 1 / (1 + math.sqrt(2))
    similarity_rescaled = (similarity - similarity_min)/(1 - similarity_min)
   # -----------------------------------------------------------
  
    # assemble results record
    results += [(
      product_a, 
      product_b, 
      size,
      similarity_rescaled
      )]
  
  # return results
  return pd.DataFrame(results)

product_pairs = (
  spark
    .table('instacart.user_ratings').filter("split='calibration'").selectExpr('user_id','product_id as product_a','normalized_purchases as ratings_a')
    .join( 
      spark.table('instacart.user_ratings').filter("split='calibration'").selectExpr('user_id','product_id as product_b','normalized_purchases as ratings_b'),
      on=['user_id'], 
      how='inner'
      )
    .filter('product_a < product_b')
    .groupBy('product_a', 'product_b')
      .agg(
          count('*').alias('size'),
          collect_list('ratings_a').alias('values_a'),
          collect_list('ratings_b').alias('values_b')
          )
    .filter('size >= 6')  # filter for minimal users
    .groupBy('product_a', 'product_b')
      .applyInPandas(
        compare_products,
        schema='''
          product_id int,
          paired_product_id int,
          size long,
          similarity double
         '''
       )
  )

# COMMAND ----------

# DBTITLE 1,Initialize Output Table
_ = spark.sql('DROP TABLE IF EXISTS instacart.product_pairs')

shutil.rmtree('/dbfs/mnt/instacart/gold/product_pairs', ignore_errors=True)

_ = spark.sql('''
  CREATE TABLE instacart.product_pairs (
    product_id int,
    paired_product_id int,
    size long,
    similarity double
    )
  USING delta
  LOCATION '/mnt/instacart/gold/product_pairs'
  ''')

# COMMAND ----------

# DBTITLE 1,Persist Product-Pairs to Disk
# persist data for future use
(
  product_pairs
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/product_pairs')
  )
  
# flip product A & product B
(
  spark
    .table('DELTA.`/mnt/instacart/gold/product_pairs`')
    .selectExpr(
      'paired_product_id as product_id',
      'product_id as paired_product_id',
      'size',
      'similarity'
      )
    .write
    .format('delta')
    .mode('append')
    .save('/mnt/instacart/gold/product_pairs')
  )
 
 # record entries for product A to product A (sim = 1.0)
(
   spark
     .table('instacart.user_ratings')
     .filter("split='calibration'")
     .groupBy('product_id')
       .agg(count('*').alias('size'))
     .selectExpr(
       'product_id as product_id',
       'product_id as paired_product_id',
       'cast(size as long) as size',
       'cast(1.0 as double) as similarity'
       )
     .write
       .format('delta')
       .mode('append')
       .save('/mnt/instacart/gold/product_pairs')
  )
   
display(
  spark.table('instacart.product_pairs')
  )

# COMMAND ----------

# MAGIC %md # Step 2a: Generate User-Based Recommendations
# MAGIC 
# MAGIC With our user-pairs assembled, we now turn generation of the actual recommendations.  Speed of retrieval is important in most recommendation scenarios.  At the same time, we need to consider how many of the users in our data set are likely to engage the recommendation engine while these data are active. When the number of users who will engage the recommendations is low relative to the total number of users in our dataset, we might consider dynamically generating (and possibly then caching for later reuse) those recommendations.
# MAGIC 
# MAGIC To dynamically generate recommendations, we might consider employing a relational database engine.  Partitioning and indexing strategies available those technologies may be employed to ensure high query-performance SLAs are met.  Denormalizing some of the data may also help in ensuring consistent retrieval speeds. 
# MAGIC 
# MAGIC We will not be demonstrating how to publish data to specific RDMBSs in this notebook as this will increase the number of dependencies for running this code.  Information and code-samples for publishing data to popular RDMBSs can be found through the following links:</p>
# MAGIC 
# MAGIC * [Azure SQL Server](https://docs.microsoft.com/en-us/sql/connect/spark/connector?view=sql-server-ver15) 
# MAGIC * [AWS RDS & other JDBC or ODBC-supported RDMBSs](https://docs.databricks.com/data/data-sources/sql-databases.html)
# MAGIC 
# MAGIC For this notebook, we will instead perform queries against Delta Lake-enabled tables.  The primary tables we'll employ are:</p>
# MAGIC 1. User-Pairs
# MAGIC 2. User-Ratings
# MAGIC 3. Products
# MAGIC 
# MAGIC We'll bring the data in these tables as follows:

# COMMAND ----------

# DBTITLE 1,Define View for User-Based Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS instacart.user_based_recommendations;
# MAGIC 
# MAGIC CREATE VIEW instacart.user_based_recommendations 
# MAGIC AS
# MAGIC   SELECT
# MAGIC     m.user_id,
# MAGIC     m.product_id,
# MAGIC     SUM(COALESCE(n.normalized_purchases, 0.0) * m.similarity) /
# MAGIC       SUM(m.similarity) as score
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.user_id,
# MAGIC       x.paired_user_id,
# MAGIC       y.product_id,
# MAGIC       x.similarity
# MAGIC     FROM instacart.user_pairs x
# MAGIC     CROSS JOIN instacart.products y
# MAGIC     ) m
# MAGIC   LEFT OUTER JOIN ( -- retrieve ratings actually provided by similar users
# MAGIC     SELECT 
# MAGIC       user_id as paired_user_id, 
# MAGIC       product_id, 
# MAGIC       normalized_purchases 
# MAGIC     FROM instacart.user_ratings
# MAGIC     WHERE split='calibration'
# MAGIC     ) n
# MAGIC     ON m.paired_user_id=n.paired_user_id AND m.product_id=n.product_id
# MAGIC   GROUP BY m.user_id, m.product_id
# MAGIC   ORDER BY score DESC

# COMMAND ----------

# MAGIC %md Let's now use our view to render recommendations for a user:

# COMMAND ----------

# DBTITLE 1,Define Cache Objects
# MAGIC %sql 
# MAGIC 
# MAGIC CACHE TABLE cached__user_ratings AS 
# MAGIC   SELECT user_id, product_id, normalized_purchases 
# MAGIC   FROM instacart.user_ratings 
# MAGIC   WHERE split='calibration';
# MAGIC   
# MAGIC CACHE TABLE cached__products AS 
# MAGIC   SELECT product_id 
# MAGIC   FROM instacart.products;

# COMMAND ----------

# DBTITLE 1,Make Recommendations
# MAGIC %sql 
# MAGIC   SELECT * 
# MAGIC   FROM instacart.user_based_recommendations 
# MAGIC   WHERE user_id = 148
# MAGIC   ORDER BY score DESC

# COMMAND ----------

# MAGIC %md  Query performance is not ideal, but again, our focus here is on logic, not performance.  To obtain the needed performance, the base tables for this query are best replicated to an RDBMS and indexed for faster access.

# COMMAND ----------

# DBTITLE 1,Release Cached Objects
# MAGIC %sql 
# MAGIC UNCACHE TABLE cached__user_ratings;
# MAGIC UNCACHE TABLE cached__products

# COMMAND ----------

# MAGIC %md # Step 2b: Generate Item-Based Recommendations
# MAGIC 
# MAGIC As with the user-based recommendations, we would typically publish several of our base tables to an RDMBS to enable the dynamic generation of recommendations.  The tables requires for this are:</p>
# MAGIC 
# MAGIC 1. Product-Pairs
# MAGIC 2. User-Ratings
# MAGIC 3. Products
# MAGIC 
# MAGIC Using these tables, we might construct a generic view to encapsulate our recommendation generation logic as follows:

# COMMAND ----------

# DBTITLE 1,Define View for Item-Based Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS instacart.item_based_recommendations;
# MAGIC 
# MAGIC CREATE VIEW instacart.item_based_recommendations 
# MAGIC AS
# MAGIC   SELECT
# MAGIC     user_id,
# MAGIC     product_id,
# MAGIC     SUM(normalized_purchases * similarity) / SUM(similarity) as score
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.user_id,
# MAGIC       y.paired_product_id as product_id,
# MAGIC       x.normalized_purchases,
# MAGIC       y.similarity,
# MAGIC       RANK() OVER (PARTITION BY x.user_id, x.product_id ORDER BY y.similarity DESC) as product_rank
# MAGIC     FROM instacart.user_ratings x
# MAGIC     INNER JOIN instacart.product_pairs y
# MAGIC       ON x.product_id=y.product_id
# MAGIC     WHERE 
# MAGIC       x.split = 'calibration'
# MAGIC     )
# MAGIC   WHERE product_rank <= 1
# MAGIC   GROUP BY user_id, product_id;

# COMMAND ----------

# MAGIC %md The view may then be queried as follows:

# COMMAND ----------

# DBTITLE 1,Define Cache Objects
# MAGIC %sql 
# MAGIC 
# MAGIC CACHE TABLE cached__user_ratings AS 
# MAGIC   SELECT user_id, product_id, normalized_purchases 
# MAGIC   FROM instacart.user_ratings 
# MAGIC   WHERE split='calibration';

# COMMAND ----------

# DBTITLE 1,Make Recommendations
# MAGIC %sql 
# MAGIC   SELECT * 
# MAGIC   FROM instacart.item_based_recommendations 
# MAGIC   WHERE user_id = 148
# MAGIC   ORDER BY score DESC

# COMMAND ----------

# DBTITLE 1,Release Cached Objects
# MAGIC %sql UNCACHE TABLE cached__user_ratings;

# COMMAND ----------

# MAGIC %md Again, performance is achieved by replicating these objects to an RDBMS.  The query here simply shows us the means by which these data are dynamically assembled to form our recommendations.
