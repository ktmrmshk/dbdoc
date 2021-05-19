# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、ユーザーベースの協調フィルタの構築に向けて、類似したユーザーを効率的に特定する方法を探ることです。このノートブックは、**Databricks 7.1+ クラスタ**で動作するように設計されています。 

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC 協調フィルタは，現代のレコメンデーション体験を実現する重要な要素です．「***あなたに似たお客様はこんなものも買っています***」というタイプのレコメンデーションは、興味を引く可能性の高い商品を特定し、購入商品の幅を広げたり、お客様のショッピングカートをより早く満たすための重要な手段となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">
# MAGIC 
# MAGIC この種のレコメンダーを構築するために必要なトランザクションデータが揃ったところで、この種のフィルターの背後にある中核的なメカニズムに目を向けてみましょう。これは、製品購入のパターンが似ている顧客を特定することが中心となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_userbasedcollab.gif" width="300">

# COMMAND ----------

# DBTITLE 1,必要なライブラリのimport
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH

from pyspark.sql.functions import col, udf, max, collect_list, lit, monotonically_increasing_id ,expr, coalesce, pow, sum
from pyspark.sql.types import *
from pyspark.sql import DataFrame

import pandas as pd
import numpy as np

import math

# COMMAND ----------

# MAGIC %md # Step 1:  評価データセット(ratings dataset)の探索
# MAGIC 
# MAGIC ユーザーベースの協調フィルタ（CF）の生成に入る前に、まずデータセットの概要を把握するために、「おすすめ」を提示する必要がある顧客の数を考えてみましょう。

# COMMAND ----------

# DBTITLE 1,レコメンドを必要とする顧客
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 'users' as entity, COUNT(DISTINCT user_id) as instances FROM instacart.user_ratings

# COMMAND ----------

# MAGIC %md To assemble a user-based recommender, we need to compare each of our roughly 200,000 users to one another. Simplistically, that equates to about 40-billion comparisons (200,000 \* 200,000).  If we consider that a comparison between user A and user B should return the same result as a comparison between user B and user A, then we can halve that number.  And if we consider that a comparison between user A and his- or herself will result in a constant, we can reduce the number of comparisons required a little further, though still about 20-billion \[(200,000 \* (200,000 - 1)) / 2\].  
# MAGIC 
# MAGIC For each user-pair, we need to compare the implicit ratings associated with each product.  In this dataset, that's about 50,000 product-level comparisons that must be performed for each user-comparison:

# COMMAND ----------

# DBTITLE 1,Products Evaluated with Each User Comparison
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT 'products', COUNT(DISTINCT product_id) FROM instacart.products

# COMMAND ----------

# MAGIC %md But most customers only buy a small number of products.  Of the potential 10-billion (200,000 \* 50,000) user-product associations possible, we only observe about 14-million within the dataset:

# COMMAND ----------

# DBTITLE 1,User-Product Ratings
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 'ratings', COUNT(*) as instances
# MAGIC FROM (
# MAGIC   SELECT DISTINCT user_id, product_id 
# MAGIC   FROM instacart.user_ratings
# MAGIC   );

# COMMAND ----------

# MAGIC %md All of this is to say that we will need to be mindful of how we might efficiently perform our user-to-user comparisons to avoid unnecessary overhead and effort.

# COMMAND ----------

# MAGIC %md # Step 2: Identify Similar Users (Brute-Force Method)
# MAGIC 
# MAGIC User-based collaborative filters are based on weighted averages constructed from the ratings of similar users. A starting point for building such a recommender is the construction of a user-to-user comparison.
# MAGIC 
# MAGIC One way to tackle such a comparison is to just do it.  In a *brute-force* exercise, each customer is compared to the others.  We can take advantage of the Databricks platform for this work by distributing the user-to-user combinations across our cluster:
# MAGIC 
# MAGIC **NOTE** We are using a LIMIT clause to restrict comparisons to a subset of only 100 users.  This is to allow is to run this code for demonstration purposes in a timely manner.

# COMMAND ----------

# DBTITLE 1,Assemble User A to User B Comparisons
ratings = (
  spark
    .sql('''
      SELECT
        user_id,
        COLLECT_LIST(product_id) as products_list,
        COLLECT_LIST(normalized_purchases) as ratings_list
      FROM (
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE 
          split = 'calibration' AND
          user_id IN (  -- constrain users to consider
            SELECT DISTINCT user_id 
            FROM instacart.user_ratings 
            WHERE split = 'calibration'
            ORDER BY user_id 
            LIMIT 100          -- limit should be > 1
            )
        )
      GROUP BY user_id
      ''')
  ).cache() # cache to eliminate overhead of reading data from disk in next steps

# assemble user A data
ratings_a = (
            ratings
              .withColumnRenamed('user_id', 'user_a')
              .withColumnRenamed('products_list', 'indices_a')
              .withColumnRenamed('ratings_list', 'values_a')
            )

# assemble user B data
ratings_b = (
            ratings
              .withColumnRenamed('user_id', 'user_b')
              .withColumnRenamed('products_list', 'indices_b')
              .withColumnRenamed('ratings_list', 'values_b')
            )

# calculate number of index positions required to hold product ids (add one to allow for index position 0)
size = spark.sql('''SELECT 1 + COUNT(DISTINCT product_id) as size FROM instacart.products''')

# join to associate user A with user B
a_to_b = (
  ratings_a
    .join(ratings_b, [ratings_a.user_a < ratings_b.user_b]) # limit user combinations to just those required
  ).crossJoin(size)

display(a_to_b)

# COMMAND ----------

# MAGIC %md With our users matched and data organized for comparison, let's convert our data into sparse vectors to enable an efficient comparison operation: 

# COMMAND ----------

# DBTITLE 1,Define Comparison Function
def compare_users( data ):
  '''
  the incoming dataset is expected to have the following structure:
     user_a, indices_a, values_a, user_b, indices_b, values_b, size
  '''
  
  # list to hold results
  results = []
  
  # retreive size (same across rows)
  size = data['size'][0]
  
  # for each entry in this subset of data ...
  for row in data.itertuples(index=False):
    
    # retrieve data from incoming dataset
    # -----------------------------------------------------------
    user_a = row.user_a
    indices_a = row.indices_a
    values_a = row.values_a
    
    user_b = row.user_b
    indices_b = row.indices_b
    values_b = row.values_b
    # -----------------------------------------------------------
    
    # construct data structures for user comparisons
    # -----------------------------------------------------------
    # assemble user A dataframe of ratings
    ind_a, val_a = zip(*sorted(zip(indices_a, values_a)))
    a = Vectors.sparse(size, ind_a, val_a)

    # assemble user B dataframe of ratings
    ind_b, val_b = zip(*sorted(zip(indices_b, values_b)))
    b = Vectors.sparse(size, ind_b, val_b)
    # -----------------------------------------------------------
    
    # calc Euclidean distance between users a and b
    # -----------------------------------------------------------
    distance = math.sqrt(Vectors.squared_distance(a, b))
   # -----------------------------------------------------------
  
    # assemble results record
    results += [(
      user_a, 
      user_b, 
      distance
      )]
  
  # return results
  return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md The function we've defined is expected to receive a pandas dataframe called *data*.  That dataframe will contain records, each of which represents a comparison between one user, *i.e.* user A, and another user, *i.e.* user B. For each user, lists of co-sorted product IDs and ratings will be provided.  The function must use these ratings to assemble the sparse vectors required for the comparison.  A simple Euclidean distance calculation is then performed to determine how *far apart* the users are from each other in terms of product ratings.
# MAGIC 
# MAGIC The function performing this work is defined as a [pandas UDF using the Spark 3.0 syntax](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#pandas-function-apis).  We will apply this function to the data organized in a Spark dataframe.  The Spark dataframe allows our data to be distributed across the worker nodes of a cluster so that the function may be applied to subsets of the dataset in a parallel manner. 
# MAGIC 
# MAGIC To ensure our data are evenly distributed across the cluster (and therefore that each worker is assigned a reasonably consistent amount of work to perform), we are calculating an id for each record in the dataset which we can then perform a modulo calculation to assign subsets of data to a distributed set. The *monotonically_increasing_id()* function does not create a perfectly sequential id value but we should expect that through the modulo operation that we will arrive at sets that contain a similar number of records.  There is overhead in this step in that we are not only calculating an id and subsetting our data against it but we are grouping our data into those subsets, triggering a shuffle operation. For small datasets, this step may add more overhead than it is worth so that you should carefully evaluate the number of records in each dataframe partition before implementing an action such as this:
# MAGIC 
# MAGIC **NOTE** The 4,950 user comparisons here are nowhere near sufficiently large to create an out of memory error on a worker of even modest size.  The row count cap is simply here should someone raise the LIMIT imposed above, exponentially increasing the number of comparisons to be performed.

# COMMAND ----------

# DBTITLE 1,Calculate User-Similarities (100 users)
# redistribute the user-to-user data and calculate similarities
similarities = (
  a_to_b
    .withColumn('id', monotonically_increasing_id())
    .withColumn('subset_id', expr('id % ({0})'.format(sc.defaultParallelism * 10)))
    .groupBy('subset_id')
    .applyInPandas(
      compare_users, 
      schema='''
        user_a int, 
        user_b int, 
        distance double
        ''')
    )

# force full execution cycle (for timings)
similarities.count()

# COMMAND ----------

# DBTITLE 1,Display Results of User Similarity Comparisons
display(
  similarities
  )

# COMMAND ----------

# DBTITLE 1,Remove Cached Dataset
# unpersist dataset that is no longer needed
_ = ratings.unpersist()

# COMMAND ----------

# MAGIC %md Before considering the timing of the brute-force comparison, take a close look at the distances between user_a and user_b in the similarities dataset above.  Notice how most values are at or close to the square root of 2. In an L2-normalized set, the maximum distance between two points should be the square root of 2. The fact that so many of our users are separated by close to that distance reflects the fact that (1) many users are truly dissimilar from one another and (2) even relatively similar users will be separated by quite a bit of distance given the cumulative effect of differences across so many product features. 
# MAGIC 
# MAGIC This later issue is often referred to as the *curse of dimensionality*.  We might reduce its effects by limiting the products we use for performing similarity calculations to a subset of fairly popular products and/or applying dimension reduction techniques (such as principal component analysis) against this data. We will not demonstrate those approaches here but this is something you might want to explore for your implementation.

# COMMAND ----------

# MAGIC %md Returning to our performance of our brute-force comparison, notice that with only 100 users our processing time is just a few seconds.  If we were to adjust our user counts keeping the number of worker nodes to a fixed size, we will see our timings increase exponentially.  Here is a chart of comparison times relative to user-counts using a 4-worker node cluster with 4 vCPUs per worker.  Notice that the time to compute grows near exponentially relative to our user count:
# MAGIC 
# MAGIC **NOTE** A scatter plot is typically used to visualize such data but the bar-chart drives how the point a little more clearly.

# COMMAND ----------

# DBTITLE 1,Seconds for User-Comparisons with Variable User Counts
timings = spark.createDataFrame(
  [
  (10,2.13),
  (10,2.31),
  (10,2.05),
  (10,1.77),
  (10,1.92),
  (100,2.62),
  (100,2.32),
  (100,2.12),
  (100,2.22),
  (100,2.32),
  (1000,138.0),
  (1000,148.0),
  (1000,150.0),
  (1000,148.0),
  (1000,151.0),
  (10000,13284.0),
  (10000,11808.0),
  (10000,12168.0),
  (10000,12392.0)
  ],
  schema='users int, seconds float'
  )

display(
  timings
)

# COMMAND ----------

# MAGIC %md The brute-force approach to user-comparisons requires that we either wait for considerable amounts of time for our calculations to complete or that we add a near exponential number of resources to the exercise to keep our timings stable.  Neither of these approaches is sustainable. And for that reason, we need to find a way to limit the comparisons perform.

# COMMAND ----------

# MAGIC %md # Step 3: Identify Similar Users (LSH Method)
# MAGIC 
# MAGIC As an alternative to a brute-force comparison, we can use a technique called [*locality-sensitive hashing* (LSH)](https://www.slaney.org/malcolm/yahoo/Slaney2008-LSHTutorial.pdf) to quickly divide our users into buckets of **potentially** similar users. The fundamental idea behind LSH is that we can limit the number of users we need to compare by placing users into buckets and limiting our comparisons to just those users in a shared bucket.  We determine which buckets users are placed into by generating random hyper-planes (represented in this 2D image as a line) and determining whether members are above or below the plane:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash00x.png' width='500'>
# MAGIC 
# MAGIC With the generation of multiple hyper-planes, we divide the users into reasonably sized buckets.  Users in a bucket are expected to be more similar to one another than members in other buckets:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash01x.png' width='500'>
# MAGIC 
# MAGIC The process doesn't guarantee a perfect separation of users into buckets where all members are more similar to each other than members in other buckets, but it provides a quick way to divvy up users into *likely* similar groups. 
# MAGIC 
# MAGIC To make use of LSH as implemented in Spark, we need to organize our user's products ratings into a sparse vector. To generate such a vector, we will create a user-defined function called *to_vector*.  We will need to pass this function not only a list of product IDs which will serve as indices in the vector, but a list of ratings for those products.  The function will also need to know how many index positions it could encounter.  As we are using the integer product IDs as our index values, we'll inform the *to_vector* UDF of the maximum product ID value in our dataset + 1 to allow a sufficient number of index positions for all of our products:

# COMMAND ----------

# DBTITLE 1,Define Function to Convert Ratings to Sparse Vector
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    
    # sort list by index (ascending)
    ind, val = zip(*sorted(zip(index_list, value_list)))
    
    return Vectors.sparse(size, ind, val)

# register function so it can be used in SQL
_ = spark.udf.register('to_vector', to_vector)

# COMMAND ----------

# MAGIC %md Now, we can prepare our dataset.  We'll do this first using SQL and then Python so that you can see multiple ways to implement the same logic. Please note that we are using the full user dataset with all 200,000+ users in it:

# COMMAND ----------

# DBTITLE 1,Prepare User Vectors (SQL)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT  -- convert lists into vectors
# MAGIC   user_id,
# MAGIC   to_vector(size, index_list, value_list) as ratings
# MAGIC FROM ( -- aggregate product IDs and ratings into lists
# MAGIC   SELECT
# MAGIC     user_id,
# MAGIC     (SELECT MAX(product_id) FROM instacart.products) + 1 as size,
# MAGIC     COLLECT_LIST(product_id) as index_list,
# MAGIC     COLLECT_LIST(normalized_purchases) as value_list
# MAGIC   FROM ( -- all users, ratings
# MAGIC     SELECT
# MAGIC       user_id,
# MAGIC       product_id,
# MAGIC       normalized_purchases
# MAGIC     FROM instacart.user_ratings
# MAGIC     WHERE split = 'calibration'
# MAGIC     )
# MAGIC   GROUP BY user_id
# MAGIC   )

# COMMAND ----------

# DBTITLE 1,Prepare User Vectors (Python)
# assemble user ratings
user_ratings = (
  spark
    .table('instacart.user_ratings')
    .filter("split = 'calibration'")
    .select('user_id', 'product_id', 'normalized_purchases')
  )

# aggregate user ratings into per-user vectors
ratings_lists = (
  user_ratings
    .groupBy(user_ratings.user_id)
      .agg(
        collect_list(user_ratings.product_id).alias('index_list'),
        collect_list(user_ratings.normalized_purchases).alias('value_list')
        )
    )

# calculate vector size
vector_size = (
  spark
    .table('instacart.products')
    .groupBy()
      .agg( 
        (lit(1) + max('product_id')).alias('size')
        )
    )

# assemble ratings dataframe
ratings_vectors = (
  ratings_lists
    .crossJoin(vector_size)
    .withColumn(
      'ratings', 
      to_vector(
        vector_size.size, 
        ratings_lists.index_list, 
        ratings_lists.value_list
        )
      )
    .select(ratings_lists.user_id, 'ratings')
  )

display(ratings_vectors)

# COMMAND ----------

# MAGIC %md We can now generate an LSH table and assign users to buckets:

# COMMAND ----------

# DBTITLE 1,Apply LSH Bucket Assignments (bucketLength=1)
bucket_length = 1
lsh_tables = 1

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# fit the algorithm to the dataset
fitted_lsh = lsh.fit(ratings_vectors)

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(ratings_vectors)
    ).cache()

# make dataset accessible via SQL
hashed_vectors.createOrReplaceTempView('hashed_vectors')

display(
  hashed_vectors
  )

# COMMAND ----------

# MAGIC %md Let's examine the hashed output to see what LSH has done. The hash field contains an array of vectors representing the bucket assignment for each table.  With one hash table, there is one vector in the array, and it's values pivot between 0 and -1. To make this easier to see, let's extract the values from that vector:
# MAGIC 
# MAGIC **NOTE** I'm going to be a little sloppy in extracting the bucket id from our hashed values vector.  Just much more succinct to do it this way.  Also, please ignore the *htable* field for now.

# COMMAND ----------

# DBTITLE 1,Display LSH Bucket Assignment
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT *
# MAGIC FROM user_comparisons

# COMMAND ----------

# MAGIC %md It appears we have users divided into two buckets.  Let's take a look at the user counts within each bucket:

# COMMAND ----------

# DBTITLE 1,User Count By Bucket
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*)
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket

# COMMAND ----------

# MAGIC %md By simply dividing our users into two buckets, we have lowered our required user-comparison count from around 20-billion to about 11-billion: 

# COMMAND ----------

# DBTITLE 1,Required User Comparisons
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md The number of buckets in the hash table is controlled by the *bucketLength* argument.  The lower the *bucketLength* value, the less space each bucket is expected to address and therefore the higher the number of buckets needed to capture all the users in the space.  You can think of this parameter as something like a reverse throttle which when lowered increases the number of hyper-planes used to carve up the space and the number of resulting buckets in the output.
# MAGIC 
# MAGIC While some notes in the [code for the BucketedRandomProjectionLSH transformation](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.scala) indicates a *bucketLength* between 1 and 10 might serve our needs best, we can see that a *bucketLength* of 1.0 is only generating a single hyper-plane which divides our users into just two buckets. We'll need to lower this parameters value to generate higher bucket counts:

# COMMAND ----------

# DBTITLE 1,Apply LSH Bucket Assignments (bucketLength=0.001)
bucket_length = 0.001
lsh_tables = 1

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# fit the algorithm to the dataset
fitted_lsh = lsh.fit(ratings_vectors)

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(ratings_vectors)
    ).cache()

# make dataset accessible via SQL
hashed_vectors.createOrReplaceTempView('hashed_vectors')

# COMMAND ----------

# DBTITLE 1,User Count by Bucket
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*) as users
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket
# MAGIC ORDER BY htable, bucket

# COMMAND ----------

# DBTITLE 1,Required User Comparisons
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md Let's start examining how reducing bucketLength has impacted our data transformations by looking at the number of users per bucket.  Instead of two broad buckets, we have something like 40 buckets into which users are placed. This reduces our required user comparison count to about 1.4-billion which is good.  But take a look at the distribution of users across these buckets. 
# MAGIC 
# MAGIC Data Scientists typically love a nice Gaussian curve, but not in this scenario.  The centering of our users around a middle bucket, *i.e.* bucket_id=-1, is another demonstration of the *curse of dimensionality*.  Simply put, a lot of our users are very similar to one another within the 50,000 product-feature hyper-space in that they reside at the edges of the space. As such, our buckets get an uneven distribution of users in them so that some buckets will contain a large number of users to compare while others have smaller numbers of users to compare.  If our goal is to reduce the number of comparisons and make sure that we can more evenly distribute those comparisons in order to take advantage of a distributed infrastructure, we'll need to be mindful of this problem.
# MAGIC 
# MAGIC But returning to *bucketLength*, we can see that lowering its value increases our bucket count. Each bucket collects the users residing in the space between the various hyper-planes that are generated to divide the overall hyper-dimensional space. Each hyper-plane is randomly generated so by increasing the bucket count, we increase the number of random hyper-planes and we increase the likelihood that two very similar users might get split into separate buckets.
# MAGIC 
# MAGIC The trick to overcoming this problem is to perform the division of users into buckets multiple times.  Each permutation will be used to generate a separate, independent *hash table*. While the problem of splitting similar users into separate buckets persists, the probability that two similar users would be repeatedly split into separate buckets across different (and independent) hash tables is lower.  And if two users reside in the same bucket across any of the hash tables generated, it becomes available for a comparison:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash03x.png' width='600'>
# MAGIC 
# MAGIC Of course, computing multiple hash tables increases the computational burden of our approach.  It also increases the number of user-comparisons that must be performed if we consider that any users found in the same bucket across all the hash tables will now be compared. Determining the right number of hash tables is all about balancing computation time with willingness to *miss* a similar user.  To explore this concept, let's keep our bucket length the same as it was in the last code block and increase our hash table count to 3:

# COMMAND ----------

# DBTITLE 1,Apply LSH Bucket Assignments (bucketLength=0.001, numHashTables=3)
bucket_length = 0.001
lsh_tables = 3

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# fit the algorithm to the dataset
fitted_lsh = lsh.fit(ratings_vectors)

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(ratings_vectors)
    ).cache()

# replace view with new hash data
hashed_vectors.createOrReplaceTempView('hashed_vectors')

display(
  hashed_vectors.select('user_id', 'hash')
  )

# COMMAND ----------

# DBTITLE 1,User Count By Hash Table & Bucket
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*) as users
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket
# MAGIC ORDER BY htable, bucket

# COMMAND ----------

# DBTITLE 1,Required User Comparisons
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md Increasing our hash table count increases the number of buckets we'll need to explore to form our user comparisons.  We can see slight differences in the distribution of our user counts between the buckets which give us a sense of the impact of all of this. In total, our required user comparisons across all three tables has risen quite a bit which would tell us a large number of customers are being split into different buckets between the three tables.
# MAGIC 
# MAGIC So, is three the right number of hash tables?  Again, the answer is not straightforward as it involves making concicous decisions about tradeoffs between processing time and results accuracy.  To see this in action around a specific user, we'll do an exhaustive, brute-force comparison for a single user, *i.e.* user_id 148, and then look at results obtained when using LSH for a fixed bucket length of 0.001 with a variable number of tables:

# COMMAND ----------

# MAGIC %md And here is the brute-force evaluation (against all 200,000 other users) for user 148:

# COMMAND ----------

# DBTITLE 1,Calculate Exhaustive Similarities for Test User
# ratings for all users
ratings = (
  spark
    .sql('''
      SELECT
        user_id,
        COLLECT_LIST(product_id) as products_list,
        COLLECT_LIST(normalized_purchases) as ratings_list
      FROM (
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE split = 'calibration'
          )
      GROUP BY user_id
      ''')
  )

# assemble user A data
ratings_a = (
            ratings
              .filter(ratings.user_id == 148) # limit to user 148
              .withColumnRenamed('user_id', 'user_a')
              .withColumnRenamed('products_list', 'indices_a')
              .withColumnRenamed('ratings_list', 'values_a')
            )

# assemble user B data
ratings_b = (
            ratings
              .withColumnRenamed('user_id', 'user_b')
              .withColumnRenamed('products_list', 'indices_b')
              .withColumnRenamed('ratings_list', 'values_b')
            )

# calculate number of index positions required to hold product ids (add one to allow for index position 0)
size = spark.sql('''SELECT 1 + COUNT(DISTINCT product_id) as size FROM instacart.products''')

# cross join to associate every user A with user B
a_to_b = (
  ratings_a
    .crossJoin(ratings_b) 
  ).crossJoin(size)

# determine number of partitions per executor to keep partition count to 100,000 records or less
partitions_per_executor = 1 + int((a_to_b.count() / sc.defaultParallelism)/100000)

# redistribute the user-to-user data and calculate similarities
brute_force = (
  a_to_b
    .withColumn('id', monotonically_increasing_id())
    .withColumn('subset_id', expr('id % ({0} * {1})'.format(sc.defaultParallelism, partitions_per_executor)))
    .groupBy('subset_id')
    .applyInPandas(
      compare_users, 
      schema='''
        user_a int, 
        user_b int, 
        distance double
        ''').cache()
    )

display(
  brute_force
    .orderBy('distance', ascending=True)
  )

# COMMAND ----------

# MAGIC %md Let's now perform LSH lookups leveraging a fixed bucket length and a variable number of hash tables:

# COMMAND ----------

# DBTITLE 1,Retrieve User 148's Vector
# retreive vector for user 148

user_148_vector = (
  ratings_a
    .crossJoin(size)
    .withColumn('vector', to_vector('size','indices_a','values_a'))
  ).collect()[0]['vector']

user_148_vector

# COMMAND ----------

# DBTITLE 1,Identify Similar Neighbors using LSH with Differing Hash Table Counts
bucket_length = 0.001

# initialize results with brute force results
results = brute_force

# initialize objects within loops
temp_lsh = []
temp_fitted_lsh = []
temp_hashed_vectors = []
temp_results = []

# loop through lsh table counts 1 through 10 ...
for i, lsh_tables in enumerate(range(1,11)):
  
  # generate lsh hashes
  temp_lsh += [
    BucketedRandomProjectionLSH(
    inputCol = 'ratings', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    )]

  temp_fitted_lsh += [temp_lsh[i].fit(ratings_vectors)]

  # calculate bucket assignments
  temp_hashed_vectors += [(
    temp_fitted_lsh[i]
      .transform(ratings_vectors)
      )]

  # lookup 100 neighbors for user 148
  temp_results += [(
    temp_fitted_lsh[i].approxNearestNeighbors(
      temp_hashed_vectors[i], 
      user_148_vector, 
      100, 
      distCol='distance_{0}'.format(str(lsh_tables).rjust(2,'0'))
      )
      .select('user_id', 'distance_{0}'.format( str(lsh_tables).rjust(2,'0')))
    )]
  
  # join results to prior results
  results = (
    results
      .join(
        temp_results[i], 
        on=results.user_b==temp_results[i].user_id, 
        how='outer'
        )
      .drop(temp_results[i].user_id)
      )

# COMMAND ----------

# MAGIC %md In the bottom of the loop in the last cell, we joined our LSH lookups to our brute-force dataset.  Let's take a look at the combined results to see what's going on with our user comparisons as we add hash tables:

# COMMAND ----------

# DBTITLE 1,Compare Exhaustive Comparisons to LSH Comparisons at Different Table Counts
display(
  results
    .orderBy('distance', ascending=True)
       )

# COMMAND ----------

# MAGIC %md Results between runs may vary due to some of the randomness associated with bucket assignment, but what you should see are instances where some more highly similar users don't appear in the LSH results until higher numbers of tables are calculated. We can affect this by increasing our bucket length, creating fewer buckets with more users in each:

# COMMAND ----------

# DBTITLE 1,Identify Similar Neighbors using LSH with Differing Hash Table Counts and Higher Bucket Length
bucket_length = 0.01

# initialize results with brute force results
results = brute_force

# initialize objects within loops
temp_lsh = []
temp_fitted_lsh = []
temp_hashed_vectors = []
temp_results = []

# loop through lsh table counts 1 through 10 ...
for i, lsh_tables in enumerate(range(1,11)):
  
  # generate lsh hashes
  temp_lsh += [
    BucketedRandomProjectionLSH(
    inputCol = 'ratings', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    )]

  temp_fitted_lsh += [temp_lsh[i].fit(ratings_vectors)]

  # calculate bucket assignments
  temp_hashed_vectors += [(
    temp_fitted_lsh[i]
      .transform(ratings_vectors)
      )]

  # lookup 100 neighbors for user 148
  temp_results += [(
    temp_fitted_lsh[i].approxNearestNeighbors(
      temp_hashed_vectors[i], 
      user_148_vector, 
      100, 
      distCol='distance_{0}'.format(str(lsh_tables).rjust(2,'0'))
      )
      .select('user_id', 'distance_{0}'.format( str(lsh_tables).rjust(2,'0')))
    )]
  
  # join results to prior results
  results = (
    results
      .join(
        temp_results[i], 
        on=results.user_b==temp_results[i].user_id, 
        how='outer'
        )
      .drop(temp_results[i].user_id)
      )
  
display(
  results
    .orderBy('distance', ascending=True)
   )

# COMMAND ----------

# MAGIC %md By altering the bucket length, we can see that we're more likely to locate similar users with fewer hash tables.  Finding the right balance between these two is more an art than a science as precision involves a tradeoff with query performance.  And an exhaustive brute-force evaluation against which you can compare these results is not viable per the explanation provided at the top of this notebook.  For this reason, users of LSH are encouraged to read [Malcom Slaney *et al.*'s in-depth exploration of these aspects of LSH tuning](https://www.slaney.org/malcolm/yahoo/Slaney2012%28OptimalLSH%29.pdf) and to develop an intuition as to how these factors come together to deliver the required results.

# COMMAND ----------

# DBTITLE 1,Clean Up Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
