# Databricks notebook source
# MAGIC %md # Introduction 
# MAGIC 
# MAGIC The purpose of this notebook is to examine how user ratings may be taken into consideration when making content-based recommendations. This notebook should be run on a **Databricks ML 7.3+ cluster**.
# MAGIC 
# MAGIC Up to this point, we've been using content-based filters to identify similar items leveraging similarities in product features.  But using user feedback, whether explicit or implicit, we can begin to construct a profile for the kinds of products we believe a customer might like and position a wider, sometimes eclectic assortment of products in the process:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_profile_recs2.png" width=600>

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.functions import col, expr, sum, count
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import Normalizer
import mlflow.spark

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: Retrieve Product Profiles
# MAGIC 
# MAGIC So far, we've made our content-based recommendations based exclusively on product features. This provides us the means of identifying items similar to the ones under consideration as may be appropriate when the goal is to enable a customer to explore product alternatives.
# MAGIC 
# MAGIC The recommender we are building here is a little different in that we will learn from product feature preferences the kinds of features likely to resonate with a customer.  This will allow us to recommend a much wider variety of products but still do so with consideration of the kinds of features likely to appeal to a customer.
# MAGIC 
# MAGIC To get started with this recommender, we will retrieve the Word2Vec features along with their cluster/bucket assignment that was generated in a prior notebook:

# COMMAND ----------

# DBTITLE 1,Retrieve Products with Features
# retrieve featurized product data
product_features = (
  spark
    .table('DELTA.`/mnt/reviews/tmp/description_sim`').selectExpr('id','bucket','w2v','features')
    .join( # join to product metadata to allow easier review of recommendations
        spark.table('reviews.metadata').select('id','asin','title','category','description'),
        on='id',
        how='inner'
      )
    .select('id', 'asin', 'w2v', 'features', 'title', 'description', 'category', 'bucket')
  )

display(
  product_features
  ) 

# COMMAND ----------

# MAGIC %md # Step 2: Assemble User Profiles
# MAGIC 
# MAGIC Our next step is to construct a weighted average of the features preferred by each customer based on their explicit product reviews.  To assist us in evaluating our recommender, we will separate reviews into separate calibration and holdout sets.  The holdout set will consist of the last two ratings provided by a user while the calibration set will consist of all ratings generated by a user prior to that:

# COMMAND ----------

# DBTITLE 1,Separate Reviews into Calibration & Holdout Sets
# retrieve user reviews numbered most recent to oldest
sequenced_reviews = (
  spark
    .sql('''
  WITH validReviews AS (
    SELECT
      reviewerID,
      product_id,
      overall as rating,
      unixReviewTime
    FROM reviews.reviews
    WHERE product_id IS NOT NULL
    )
  SELECT
    a.reviewerID,
    a.product_id,
    a.rating,
    row_number() OVER (PARTITION BY a.reviewerID ORDER BY a.unixReviewTime DESC) as seq_id
  FROM validReviews a
  LEFT SEMI JOIN (SELECT reviewerID FROM validReviews GROUP BY reviewerID HAVING COUNT(*)>=5) b
    ON a.reviewerID=b.reviewerID
    ''')
  )

# get last two ratings as holdout
reviews_hold = (
  sequenced_reviews
    .filter('seq_id <= 2')
    .select('reviewerID', 'product_id', 'rating')
  )

# get all but last two ratings as calibration
reviews_cali = (
  sequenced_reviews
    .filter('seq_id > 2')
    .select('reviewerID', 'product_id', 'rating')
  )

display(
  reviews_cali
  )

# COMMAND ----------

# MAGIC %md Now we can assemble averages of the features of the products that individual users rated, using those ratings as weights.  In doing so, we need to give careful consideration to the ratings, presented here on a scale of 1 to 5.  With most users, the vast majority of purchases on a site like Amazon go unrated.  A typical user will give a rating when they are extremely pleased or extremely dissatisfied with a product so that we might expect ratings to be skewed a bit towards the ends of the scale:

# COMMAND ----------

display(
  reviews_cali
    .groupBy('rating')
      .agg( count('*').alias('instances'))
    .orderBy('rating')
  )

# COMMAND ----------

# MAGIC %md In addition to possibly missing ratings in the middle of our scale, we need to consider what the ratings mean.  Do they represent preferences or are they a response to the product or the supplier? If the customer was interested enough in the product to make the purchase, that probably is a more accurate indicator of preference.  That they were disappointed with the product probably means it didn't live up to expectations that otherwise did connect with the customer's interests. If we put this notion aside and accept our ratings as expressions of preferences, then ratings of 1 and 2 are clear indicators of a lack of preference, but what about a 3?  If a 3 represents a baseline for expression of preference, should we use ratings below 3 to push our averages in a negative direction?
# MAGIC 
# MAGIC How we address these concerns is highly dependent on the specific business context surrounding the ratings. Here, we'll limit our development of user profiles to just those instances where the customer expressed strong preferences using ratings of 4 and 5, ignoring all other ratings.  And we'll leave the 4 and 5 ratings scaled as they are.  Again, these may not be the right choices for your business.
# MAGIC 
# MAGIC To perform the weighted averaging against a feature vector, we have two choices.  We can construct a custom function and apply it to our vectors or we can use the [Summarizer transformer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.stat.Summarizer). The Summarizer transformer allows us to perform simple aggregations against vectors with little code and includes support for weighted means.  Notice that for this, we are making use of our Word2Vec features in their pre-normalized state which means we'll need to apply normalization after they are averaged:
# MAGIC 
# MAGIC **NOTE** Other approaches such as taking the maximum of a feature value may also be used depending on your needs.

# COMMAND ----------

# DBTITLE 1,Assemble User-Profiles
# calculate weighted averages on product features for each user
user_profiles_cali = (
  product_features
    .join(
      reviews_cali.filter('rating >= 4'),  # limit ratings to 4s and 5s as discussed above
      on=[product_features.id==reviews_cali.product_id], 
      how='inner'
      )
    .groupBy('reviewerID')
      .agg( 
        Summarizer.mean(col('w2v'), weightCol=col('rating')).alias('w2v')  
        )
  )

user_profiles_cali_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(user_profiles_cali)
  ).cache()


display(
  user_profiles_cali_norm
  )

# COMMAND ----------

# MAGIC %md We now have one feature vector for each user representing the weighted feature preferences for that user.  We can think of this vector as representing the *ideal product* for each users (based on his or her feedback).  Our goal will now be to find products similar to this *ideal* and for that, we'll need to assign each feature vector to a cluster/bucket. The bucketed profiles are persisted for later use:
# MAGIC 
# MAGIC **NOTE** We are re-using the clustering model persisted in a prior notebook. Retrieving this model from the mlflow registry triggers a warning message which is verbose but which can be ignored.
# MAGIC 
# MAGIC **NOTE** We encountered some difficulties writing this one dataset to storage.  We are still investigating but for now we're persisting the data with Parquet.

# COMMAND ----------

# DBTITLE 1,Assign Cluster/Bucket & Persist User-Profiles
# retrieve model from mlflow
cluster_model = mlflow.spark.load_model(
    model_uri='models:/description_clust/None'
  )

# assign profiles to clusters/buckets
user_profiles_cali_clustered = (
  cluster_model.transform(user_profiles_cali_norm)
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/gold/user_profiles_cali', ignore_errors=True)

# persist dataset as delta table
(
  user_profiles_cali_clustered
   .write
   .format('parquet')
   .mode('overwrite')
   .partitionBy('bucket')
   .save('/mnt/reviews/gold/user_profiles_cali')
  )

display(
  spark.table('PARQUET.`/mnt/reviews/gold/user_profiles_cali`')
  )

# COMMAND ----------

# DBTITLE 1,Examine Distribution by Bucket
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as profiles
# MAGIC FROM PARQUET.`/mnt/reviews/gold/user_profiles_cali`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# MAGIC %md # Step 3: Build & Evaluate Recommendations
# MAGIC 
# MAGIC We now have our products with their various features and our user-profiles representing product feature preferences.  To find products we might recommend, we'll simply calculate similarities between the products and these user-preferences.
# MAGIC 
# MAGIC To enable evaluation of these recommendations, we'll limit our users to a small random sample and calculate a weighted mean percent score much like we did with our collaborative filters. For more information on that evaluation method, please see the relevant notebook:

# COMMAND ----------

# DBTITLE 1,Define Function for Distance Calculation
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors}
# MAGIC 
# MAGIC val euclidean_distance = udf { (v1: Vector, v2: Vector) =>
# MAGIC     sqrt(Vectors.sqdist(v1, v2))
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("euclidean_distance", euclidean_distance)

# COMMAND ----------

# DBTITLE 1,Take Random Sample of Reviewers
cali_profiles= spark.table('PARQUET.`/mnt/reviews/gold/user_profiles_cali`').sample(False, 0.01)

cali_profiles.count()

# COMMAND ----------

# DBTITLE 1,Determine Recommendations for Selected Users
# make recommendations for sampled reviewers
recommendations = (
  product_features
    .hint('skew','bucket') # hint to ensure join is balanced
    .withColumnRenamed('features', 'features_b')
    .join( cali_profiles.withColumnRenamed('features', 'features_a'), on='bucket', how='inner') # join products to profiles on buckets
    .withColumn('distance', expr('euclidean_distance(features_a, features_b)')) # calculate similarity
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select('reviewerID', 'id', 'similarity')
    .withColumn('rank_ui', expr('percent_rank() OVER(PARTITION BY reviewerID ORDER BY similarity DESC)')) # calculate percent rank for recommendations
    )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/gold/user_profile_recommendations', ignore_errors=True)

# persist dataset as delta table
(
  recommendations
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/gold/user_profile_recommendations')
  )

# COMMAND ----------

# DBTITLE 1,Show Selection of Results for One Reviewer
# we are retrieving a subset of recommendations for one user so that the range of rank_ui values is more visible
display(
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`')
    .join( spark.table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`').limit(1), on='reviewerID', how='left_semi' )
    .sample(False,0.01) 
    .orderBy('rank_ui', ascending=True)
  )

# COMMAND ----------

# MAGIC %md And now we can see where in the recommendations our customer makes their next purchases:

# COMMAND ----------

# retreive recommendations generated in prior step
recommendations = spark.table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`')

# calculate evaluation metric
display(
  reviews_hold
  .join(
    recommendations, 
    on=[reviews_hold.product_id==recommendations.id, reviews_hold.reviewerID==recommendations.reviewerID], 
    how='inner'
    )
  .withColumn('weighted_r', recommendations.rank_ui * reviews_hold.rating)
  .groupBy()
    .agg( sum('weighted_r').alias('numerator'),
          sum('rating').alias('denominator')
        )
  .withColumn('mean_percent_rank', expr('numerator / denominator'))
  .select('mean_percent_rank')
  )

# COMMAND ----------

# MAGIC %md Relative to our collaborative filters, the mean percent rank score presented here is not as strong and the likely culprit is the breadth of products for which we are attempting to make recommendations here. With so many products in the mix, customers who buy diverse set of products likely have a range of preferences that are being aggregated to form a single user profile.  To give an example, I have certain preferences for novelty and high-quality in some product categories but very different preferences for lower cost and consistency/reliability in other categories. It might make more sense to build different profiles for someone like me, aligned with specific product categories. Still, the base techniques for applying ratings to content-based recommenders should be clear and adaptable to these or other scenarios. 
