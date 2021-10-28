# Databricks notebook source
# MAGIC %md # Introduction 
# MAGIC 
# MAGIC The purpose of this notebook is to examine techniques which may be applied to the various product metadata fields in order to calculate the similarities that will form the basis of our recommendations. Focused on category hierarchy data, this serves to as a continuation of feature extraction work performed across the last two notebooks. Rarely would recommendations based on this kind of data be used on its own but instead in combination with other product recommenders to adjust scores where the need is to constrain recommendations around the product hierarchy:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC This notebook should be run on a **Databricks ML 7.3+ cluster**.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, lit, count, countDistinct, col, array_join, expr, monotonically_increasing_id, explode

from typing import Iterator

from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH, HashingTF, Word2Vec
from pyspark.ml.clustering import LDA, KMeans, BisectingKMeans
from pyspark.ml.linalg import Vector, Vectors

from pyspark.sql import DataFrame

import mlflow.spark

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: Retrieve & Prepare Category Data
# MAGIC 
# MAGIC In an ideal scenario, our product hierarchy would be organized in a consistent manner such that every entry rolled up to a shared root node.  Such a structure, known as a tree, would allow us to calculate product similarities based on the position of products within the tree structure.  [Thabet Slimani](https://arxiv.org/ftp/arxiv/papers/1310/1310.8059.pdf) provides an **excellent** review of various similarity metrics that could be used with trees.
# MAGIC 
# MAGIC Unfortunately, the product category captured in this dataset does not form a tree.  Instead, we find inconsistencies where a child category might be positioned under a parent category with one product with their positions reversed in another or where a child category might leap over a parent to roll directly into a shared ancestor.  The problems with the category hierarchy in this dataset are not easily corrected so that we cannot treat this data as a tree.  That said, we can still apply a little creativity to assess similarity.
# MAGIC 
# MAGIC **NOTE** If this describes a dataset that your organization manages, investing in a master data management solution may help you move the data into a more workable structure over time.
# MAGIC 
# MAGIC To get us started with our category data, let's first retrieve a few values and examine its structure:

# COMMAND ----------

# DBTITLE 1,Retrieve Category Data
# retrieve descriptions for each product
categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category) > 0')
    .select('id', 'asin', 'category')
  )

num_of_categories = categories.count()
num_of_categories

# COMMAND ----------

display(
  categories
  )

# COMMAND ----------

# MAGIC %md The category data is organized as an array where the array index indicates the level in the hierarchy.  Appending ancestry information to each level's name allows us to uniquely identify each level within the overall structure so that categories with a common name but residing under different parents or other ancestors may be distinguishable from one another. We'll tackle this using a pandas UDF:
# MAGIC 
# MAGIC **NOTE** We are finding many levels where the level name is more like a product or feature description.  We're not sure if this is valid data or represents a misparsing of data from the source website.  We'll limit the category hierarchy to 10 levels max and break the hierarchy should we encounter a level name with more than 100 characters to avoid this data.  These are arbitrary settings that you may want to adjust or discard for your dataset.

# COMMAND ----------

# DBTITLE 1,Rename Categories to Include Ancestry
@pandas_udf(ArrayType(StringType()))
def cleanse_category(array_series: pd.Series) -> pd.Series:
  
  def cleanse(array):
    delim = '|'
    level_name = ''
    ret = []
    
    for a in array[:10]:  # limit to 10 levels
      if len(a) > 100:     # stop if level name more than max chars
        break
      else:
        level_name += a.lower() + '|'
        ret += [level_name]
    return ret
          
  return array_series.apply(cleanse)


categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category)>0')
    .select('id','asin','category')
    .withColumn('category_clean', cleanse_category('category'))
  )

display(categories)

# COMMAND ----------

# MAGIC %md # Step 2: Calculate Category Similarities
# MAGIC 
# MAGIC With our level names adjusted, let's one-hot encode the category levels for each product.  We'll use the CountVectorizer used with our TF-IDF calculations to tackle this, setting it's *binary* argument to *True* so that all output is either 0 or 1 (which it should be anyway).  As before, this transform will limit the entries to some top number of frequently occurring values.  We might adjust this from its default based on our knowledge of the product categories but for now we'll leave it as-is:

# COMMAND ----------

# DBTITLE 1,One-Hot Encode the Categories
categories_encoded = (
  HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )
    .transform(categories)
    .select('id','asin','category','category_onehot')
    .cache()
  )

display(
  categories_encoded
  )

# COMMAND ----------

# MAGIC %md As before, we need to divide our records into buckets.  Because of how we have assembled our categories, we know that if two items don't have the same top-level category, they should not have any other feature overlap. So, let's group our products into buckets based on the first member of the categories array and see where that leads us:

# COMMAND ----------

# DBTITLE 1,Group Items Based on Top Parent
roots = (
  categories_encoded
    .withColumn('root', expr('category[0]'))
    .groupBy('root')
      .agg(count('*').alias('members'))
    .withColumn('bucket', monotonically_increasing_id())
  )

categories_clustered = (
  categories_encoded
    .withColumn('root', expr('category[0]'))
    .join(roots, on='root', how='inner')
    .select('id','asin','category','category_onehot','bucket')
  )

display(roots.orderBy('members'))

# COMMAND ----------

# MAGIC %md Approaching the problem this way does give use some skewed results but this seems reasonably manageable without further transformation.
# MAGIC 
# MAGIC Now we can perform a simple calculation for similarity.  Unlike our other features where similarity was measured in terms of distance (or angle), these features can be compared using a simple [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) score where we divide the number of overlapping levels by the distinct number of levels between two products:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_jaccard_similarity2.png" width="250">
# MAGIC 
# MAGIC As before, we'll perform this work using a Scala function which we will register with Spark SQL so that we might make use of it against our DataFrame defined in Python:

# COMMAND ----------

# DBTITLE 1,Define Function for Jaccard Similarity Calculation
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.SparseVector
# MAGIC 
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)

# COMMAND ----------

# MAGIC %md Let's return to our sample product to see how this is working:
# MAGIC 
# MAGIC **NOTE** We are using the same sample product as was used in the prior notebooks.

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Product
sample_product = categories_clustered.filter("asin=='B01G6M7CLK'")
display(sample_product)

# COMMAND ----------

# DBTITLE 1,Find Similar Products
display(
  categories_clustered
    .withColumnRenamed('category_onehot', 'features_b')
    .join(sample_product.withColumnRenamed('category_onehot', 'features_a'), on='bucket', how='inner')
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .orderBy('similarity', ascending=False)
    .limit(100)
    .select(categories_clustered.id, categories_clustered.asin, categories_clustered.category, 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md # Step 3: Combine Similarity Scores to Build Recommendations
# MAGIC 
# MAGIC At this point we have the ability to compare products for similarity based on their titles, descriptions and category assignment.  Individually, each feature set could be used to make a recommendation but we might find that combining information derived from multiple elements gives us results better aligned with our goals.  For example, we might find that title-based recommendations are a little to *literal* while description-based recommendations are a little too *vague* but together they might give us the right balance to make compelling recommendations or that using one or both of these feature sets to calculate similarities can be combined with a category similarity metric as shown here to prefer product recommendations from within the same or related parts of the category hierarchy.
# MAGIC 
# MAGIC In each of these scenarios we have to consider how best to combine our similarity metrics. One simple choice is to simply multiply each similarity score against each other to form a simple combined score. Another might be to weight each similarity score and then add the weighted values to each other. With a little creativity, we could come up with still more approaches to using similarity scores generated by multiple recommenders to arrive at a unified set of recommendations.  The point in considering how might combine these inputs is not to specify one way to proceed but instead to highlight that depending on your goals, any number of approaches may be reasonable.
