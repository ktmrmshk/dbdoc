# Databricks notebook source
# MAGIC %md The purpose of this notebook is to build and evaluate item-based collaborative filtering recommendations.  This notebook is designed to run on a **Databricks 7.1+ cluster**.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.sql.functions import col, udf, max, collect_list, lit, monotonically_increasing_id ,expr, coalesce, pow, sum, count
from pyspark.sql.types import *
from pyspark.sql import DataFrame

import pandas as pd
import numpy as np

import math

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: Build Product Comparisons Dataset
# MAGIC 
# MAGIC When we constructed our user-based collaborative filter, we built a vector for each user representing the implied ratings across all nearly 50,000 products in the product catalog.  These vectors would serve as the basis for calculating similarities between users.  With about 200,000 users in the system, this resulted in about 20-billion potential user comparisons which we short-cutted using Locale Sensitivity Hashing.
# MAGIC 
# MAGIC But consider that the only way a recommendation between two users could be made is if a given customer bought products A and B and the other customer bought either product A or B.  This provides us another way to approach the problem of using user-derived ratings, one that limits the number of comparisons by focusing on points of overlap between users:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_itembased.gif" width="300">

# COMMAND ----------

# DBTITLE 1,Adjust Shuffle Partitions for Performance
_ = spark.conf.set('spark.sql.shuffle.partitions',sc.defaultParallelism * 100)

# COMMAND ----------

# MAGIC %md Let's get started by examining the number of product pairs in our dataset:

# COMMAND ----------

# DBTITLE 1,Cache Ratings Data for Performance
# MAGIC %sql  CACHE TABLE instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,Unique Product Co-Occurrences
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   COUNT(*) as comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC      a.product_id as product_a,
# MAGIC      b.product_id as product_b,
# MAGIC      COUNT(*) as users
# MAGIC   FROM instacart.user_ratings a
# MAGIC   INNER JOIN instacart.user_ratings b
# MAGIC     ON  a.user_id = b.user_id AND 
# MAGIC         a.split = b.split
# MAGIC   WHERE a.product_id < b.product_id AND
# MAGIC         a.split = 'calibration'
# MAGIC   GROUP BY a.product_id, b.product_id
# MAGIC   HAVING COUNT(*) > 1 -- exclude purchase combinations found in association with only one user
# MAGIC   )    

# COMMAND ----------

# MAGIC %md While our product catalog consists of nearly 50,000 products which in theory could supply us 1.25-billion unique product pairs, the actual number of co-occurrences observed (where there is more than one customer involved) is closer to 56-million, less than a tenth of our theoretical number. By focusing on product pairs that actually occur, we limit our calculations to those that have the potential to be relevant and greatly reduce the complexity of our problem. This is the [core insight](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) behind item-based collaborative filtering.
# MAGIC 
# MAGIC But how exactly should we compare products in this scenario? In our previous collaborative filter, we built a feature vector containing an entry for each of our nearly 50,000 products.  If we flip the structure around, should we build a feature vector with an entry for each of our 200,000+ users?
# MAGIC 
# MAGIC The short answer is, *No*.  The longer answer is that a particular product comparison is performed because a user has purchased both products in a pair.  As a result, each users associated with a product pair contributes an implied rating to each side of the evaluation. But most product pairs have a limited number of customers associated with it:

# COMMAND ----------

# DBTITLE 1,Number of Users Associated with Product Pairs
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   users,
# MAGIC   COUNT(*) as occurances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC      a.product_id as product_a,
# MAGIC      b.product_id as product_b,
# MAGIC      COUNT(*) as users
# MAGIC   FROM instacart.user_ratings a
# MAGIC   INNER JOIN instacart.user_ratings b
# MAGIC     ON  a.user_id = b.user_id AND 
# MAGIC         a.split = b.split
# MAGIC   WHERE a.product_id < b.product_id AND
# MAGIC         a.split = 'calibration'
# MAGIC   GROUP BY a.product_id, b.product_id
# MAGIC   HAVING COUNT(*) > 1       -- exclude purchase combinations found in association with only one user
# MAGIC   )
# MAGIC GROUP BY users

# COMMAND ----------

# MAGIC %md It is suprising that the highest number of users who have purchased a given product combination is less than 30,000 and that only occurs once.  If we are concerned about extreme cases such as this, we could limit the user ratings considered to a random sampling of all the available ratings once the number of users associated with a pair exceeds a certain ratio. (This idea comes directly from the Amazon paper referenced above.)  Still, most combinations occur only between a very small number of users so that for each pair we simply need to construct a feature vector of a modest size in order to measure product similarities.  
# MAGIC 
# MAGIC This leads to an insteresting question: should we consider product pairs associated with just a few users?  If we include combinations purchased by only 2, 3 or some other trivially small number of users, do we start introducing products into the recommendations that might not be commonly considered? Depending on our goals, the inclusion of unusual product combinations may be a good thing or may be a bad thing.  Dealing with groceries, where novelty and suprise are not typically the goal, it seems to make sense that we might exclude products with too few co-occurances.  Later, we'll work to determine exactly what that the cutoff should be, but for now, let's construct our product vectors so that we might proceed with the exercise:

# COMMAND ----------

# DBTITLE 1,Calculate Product Similarities
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
      distance,
      similarity_rescaled
      )]
  
  # return results
  return pd.DataFrame(results)

# assemble user ratings for each product in comparison
product_comp = (
  spark
    .sql('''
      SELECT
         a.product_id as product_a,
         b.product_id as product_b,
         COUNT(*) as size,
         COLLECT_LIST(a.normalized_purchases) as values_a,
         COLLECT_LIST(b.normalized_purchases) as values_b
      FROM instacart.user_ratings a
      INNER JOIN instacart.user_ratings b
        ON  a.user_id = b.user_id AND 
            a.split = b.split
      WHERE a.product_id < b.product_id AND
            a.split = 'calibration'
      GROUP BY a.product_id, b.product_id
      HAVING COUNT(*) > 1
    ''')
  )

# calculate product simularities
product_sim = (
  product_comp
    .withColumn('id', monotonically_increasing_id())
    .withColumn('subset_id', expr('id % ({0})'.format(sc.defaultParallelism * 10)))
    .groupBy('subset_id')
    .applyInPandas(
      compare_products, 
      schema='''
        product_a int,
        product_b int,
        size int,
        distance double,
        similarity double
        '''
      )
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/instacart/gold/product_sim', ignore_errors=True)

# persist data for future use
(
  product_sim
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/product_sim')
  )

display(
  spark.table('DELTA.`/mnt/instacart/gold/product_sim`')
  )

# COMMAND ----------

# MAGIC %md At this point, our work in only about half complete.  We've constructed pairs for product A and product B but excluded the product B to product A pairs as well as the product A to product A pairs. (The a.product_id < b.product_id portion of the queries above are where this occurs.)  Let's insert these into our data set now:

# COMMAND ----------

# DBTITLE 1,Append Skipped Product Comparisons
# flip product A & product B
(
  spark
    .table('DELTA.`/mnt/instacart/gold/product_sim`')
    .selectExpr(
      'product_b as product_a',
      'product_a as product_b',
      'size',
      'distance',
      'similarity'
      )
    .write
    .format('delta')
    .mode('append')
    .save('/mnt/instacart/gold/product_sim')
  )

# COMMAND ----------

# DBTITLE 1,Append Product Self-Comparisons
# record entries for product A to product A (sim = 1.0)
(
  spark
    .table('instacart.user_ratings')
    .filter("split='calibration'")
    .groupBy('product_id')
      .agg(count('*').alias('size'))
    .selectExpr(
      'product_id as product_a',
      'product_id as product_b',
      'cast(size as int) as size',
      'cast(0.0 as double) as distance',
      'cast(1.0 as double) as similarity'
      )
    .write
      .format('delta')
      .mode('append')
      .save('/mnt/instacart/gold/product_sim')
  )

# COMMAND ----------

# DBTITLE 1,Uncache Ratings Data
# MAGIC %sql  UNCACHE TABLE instacart.user_ratings

# COMMAND ----------

# MAGIC %md By rethinking the problem, we've enabled a more direct approach to our comparison challenge.  But how exactly do we make recommendations using this data structure?

# COMMAND ----------

# MAGIC %md # Step 2: Build Recommendations
# MAGIC 
# MAGIC With our user-based collaborative filter, we generated recommendations by calculating a weighted average of user-ratings extracted from similar users.  Here, we'll retrieve the products purchased by a user in our calibration period. Those products will be used to retrieve all product pairs where product A is one of the products in our purchased set.  Implied ratings and similarity scores will again be used to construct weighted averages which will serve as recommendation scores and a percent rank will then be calculated to sequence the recommendations.  We'll demonstrate this here for a single user, *user_id 148*:

# COMMAND ----------

# DBTITLE 1,Cache Ratings for Performance
# MAGIC %sql  CACHE TABLE instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,Calculate Recommendations for a Sample User
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   user_id,
# MAGIC   product_id,
# MAGIC   recommendation_score,
# MAGIC   PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     y.product_b as product_id,
# MAGIC     SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score
# MAGIC   FROM instacart.user_ratings x
# MAGIC   INNER JOIN DELTA.`/mnt/instacart/gold/product_sim` y
# MAGIC     ON x.product_id=y.product_a
# MAGIC   WHERE 
# MAGIC     x.split = 'calibration' AND x.user_id=148
# MAGIC   GROUP BY x.user_id, y.product_b
# MAGIC   )
# MAGIC ORDER BY user_id, rank_ui

# COMMAND ----------

# MAGIC %md As before, we have recommendations but are they any good? Let's evaluate our recommendations against the evaluation data using the same mean percent rank evaluation metric employed in the last notebook.  As before, we'll limit our evaluation to a 10% sample of our all users for expediency:

# COMMAND ----------

# DBTITLE 1,Define Random Selection of Users
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS random_users;
# MAGIC 
# MAGIC CREATE TEMP VIEW random_users 
# MAGIC AS
# MAGIC   SELECT user_id
# MAGIC   FROM (
# MAGIC     SELECT DISTINCT 
# MAGIC       user_id
# MAGIC     FROM instacart.user_ratings
# MAGIC     ) 
# MAGIC   WHERE rand() <= 0.10;
# MAGIC   
# MAGIC CACHE TABLE random_users;
# MAGIC 
# MAGIC SELECT * FROM random_users;

# COMMAND ----------

# DBTITLE 1,Calculate Evaluation Metric without Constraints
eval_set = (
  spark
    .sql('''
    SELECT 
      m.user_id,
      m.product_id,
      m.r_t_ui,
      n.rank_ui
    FROM (
      SELECT
        user_id,
        product_id,
        normalized_purchases as r_t_ui
      FROM instacart.user_ratings 
      WHERE split = 'evaluation' -- the test period
        ) m
    INNER JOIN (
      SELECT
        user_id,
        product_id,
        recommendation_score,
        PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
      FROM (
        SELECT
          x.user_id,
          y.product_b as product_id,
          SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score
        FROM instacart.user_ratings x
        INNER JOIN DELTA.`/mnt/instacart/gold/product_sim` y
          ON x.product_id=y.product_a
        INNER JOIN random_users z
          ON x.user_id=z.user_id
        WHERE 
          x.split = 'calibration'
        GROUP BY x.user_id, y.product_b
        )
      ) n
      ON m.user_id=n.user_id AND m.product_id=n.product_id
      ''')
  )

display(
  eval_set
    .withColumn('weighted_r', col('r_t_ui') * col('rank_ui') )
    .groupBy()
      .agg(
        sum('weighted_r').alias('numerator'),
        sum('r_t_ui').alias('denominator')
        )
    .withColumn('mean_percent_rank', col('numerator')/col('denominator'))
    .select('mean_percent_rank')
  )

# COMMAND ----------

# MAGIC %md **Wow!** Our evaluation score isn't just bad, it's worse than if we had made random suggestions.  *How could this be?!*
# MAGIC 
# MAGIC The most likely reason is that we are leveraging all product combinations with no consideration of how many users may have actually purchased the two products that form a product pair (as discussed earlier in this notebook). If a combination happens to be highly rated for a very small number of users, it might shoot to the top of our rankings while products more popular (and therefore exposed to a wider range of ratings) might be pushed below it.  With this in mind, we might limit the products we recommend to those with a minimum number of user ratings associated with a product pair.
# MAGIC 
# MAGIC In addition, we might recognize that even with a user-minimum, we may still have a large number of products to recommend.  If we limit our recommendations to some maximum number of top ranked products, we might further improve our ratings.  It's important to remember that our dataset includes product-self recommendations, *i.e.* product A is most similar to product A, so that when we set this limit we want to keep in mind that one recommendation is already baked in.
# MAGIC 
# MAGIC Let's see how adjustments to user-minimums and product maximums affect our evaluation metric:
# MAGIC 
# MAGIC **NOTE** This step will take a while to run depending on the resources allocated to your cluster.

# COMMAND ----------

# DBTITLE 1,Iterate over Thresholds
_ = spark.sql("CACHE TABLE instacart.user_ratings")
_ = spark.sql("CACHE TABLE DELTA.`/mnt/instacart/gold/product_sim`")

results = []

for i in range(1,21,1):
  print('Starting size = {0}'.format(i))
  
  for j in [2,3,5,7,10]:
  
    rank = (
        spark
          .sql('''
            SELECT
              SUM(r_t_ui * rank_ui) / SUM(r_t_ui) as mean_percent_rank
            FROM (
              SELECT 
                m.user_id,
                m.product_id,
                m.r_t_ui,
                n.rank_ui
              FROM (
                SELECT
                  user_id,
                  product_id,
                  normalized_purchases as r_t_ui
                FROM instacart.user_ratings 
                WHERE split = 'evaluation' -- the test period
                  ) m
              INNER JOIN (
                SELECT
                  user_id,
                  product_id,
                  recommendation_score,
                  PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
                FROM (
                  SELECT
                    user_id,
                    product_id,
                    SUM(normalized_purchases * similarity) / SUM(similarity) as recommendation_score
                  FROM (
                    SELECT
                      x.user_id,
                      y.product_b as product_id,
                      x.normalized_purchases,
                      y.similarity,
                      RANK() OVER (PARTITION BY x.user_id, x.product_id ORDER BY y.similarity DESC) as product_rank
                    FROM instacart.user_ratings x
                    INNER JOIN DELTA.`/mnt/instacart/gold/product_sim` y
                      ON x.product_id=y.product_a
                    LEFT SEMI JOIN random_users z
                      ON x.user_id=z.user_id
                    WHERE 
                      x.split = 'calibration' AND
                      y.size >= {0}
                    )
                  WHERE product_rank <= {1}
                  GROUP BY user_id, product_id
                  )
                ) n
                ON m.user_id=n.user_id AND m.product_id=n.product_id
              )
            '''.format(i,j)
              ).collect()[0]['mean_percent_rank']
        )

    results += [(i, j, rank)]

  
display(
  spark
    .createDataFrame(results, schema="min_users int, max_products int, mean_percent_rank double")
    .orderBy('min_users','max_products')
  )

# COMMAND ----------

# MAGIC %md It appears that constraining our recommendations a bit leads to better results.  We might consider moving these constraints into the construction of the product comparisons dataset to limit our ETL cycles and improve query performance.  We are not performing at the levels we saw with our user-based collaborative filters, but the results are not bad (and may even be improved with further adjustments to our logic).

# COMMAND ----------

# DBTITLE 1,Clear Cached Data
# MAGIC %sql  
# MAGIC UNCACHE TABLE instacart.user_ratings;
# MAGIC UNCACHE TABLE random_users;
# MAGIC UNCACHE TABLE DELTA.`/mnt/instacart/gold/product_sim`;
