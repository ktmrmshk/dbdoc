# Databricks notebook source
# MAGIC %md # Introduction
# MAGIC 
# MAGIC The purpose of this notebook is to examine how features may be extracted from product descriptions in order to calculate similarities between products. (Other metadata fields are explored in the notebooks that accompany this one.) These similarities will be used as the basis for making **Related products** recommendations:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC This notebook should be run on a **Databricks ML 7.3+ cluster**.

# COMMAND ----------

# MAGIC %md **NOTE** The cluster with which this notebook is run should be created using a [cluster-scoped initialization script](https://docs.databricks.com/clusters/init-scripts.html?_ga=2.158476346.1681231596.1602511918-995336416.1592410145#cluster-scoped-init-script-locations) which installs the NLTK WordNet corpus and Averaged Perceptron Tagger.  The following cell can be used to generate such a script but you must associate it with the cluster before running any code that depends upon it:

# COMMAND ----------

# DBTITLE 1,Generate Cluster Init Script
dbutils.fs.mkdirs('dbfs:/databricks/scripts/')

dbutils.fs.put(
  '/databricks/scripts/install-nltk-downloads.sh',
  '''#!/bin/bash\n/databricks/python/bin/python -m nltk.downloader wordnet\n/databricks/python/bin/python -m nltk.downloader averaged_perceptron_tagger''', 
  True
  )

# alternatively, you could install all NLTK elements with: python -m nltk.downloader all

# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/install-nltk-downloads.sh')
  )

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

import nltk

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: Prepare Description Data
# MAGIC 
# MAGIC Titles are useful, but carry limited information.  Longer descriptions (as found in the *description* field of our dataset) provide quite a bit more detail which we might use to identify similar items.
# MAGIC 
# MAGIC But having more words with which to make a comparison means more data to process and more complexity to deal with.  To help us with that, we can employ one of several dimension reduction techniques which attempt to condense text into a much more narrow set of *topics* or *concepts* which can then be used as the basis for comparison.
# MAGIC 
# MAGIC To get started exploring this direction, let's flatten the array that holds our description data and tokenize the words in the text as we did earlier:

# COMMAND ----------

# DBTITLE 1,Retrieve Descriptions
# retrieve descriptions for each product
descriptions = (
  spark
    .table('reviews.metadata')
    .filter('size(description) > 0')
    .withColumn('descript', array_join('description',' '))
    .selectExpr('id', 'asin', 'descript as description')
  )

num_of_descriptions = descriptions.count()
num_of_descriptions

# COMMAND ----------

# present descriptions
display(
  descriptions
  )

# COMMAND ----------

# DBTITLE 1,Retrieve Words from Descriptions
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='description', 
    outputCol='words'
    )

description_words = tokenizer.transform(descriptions)

display(description_words)

# COMMAND ----------

# MAGIC %md # Step 2a: Extract LDA Topic Features
# MAGIC 
# MAGIC Let's now explore how we might use [Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158) to reduce our descriptions to a condensed set of topics:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_LDA.gif" width="500">
# MAGIC 
# MAGIC While the math behind LDA can get complex, the technique itself is fairly easy to understand.  In a nutshell, we examine the co-occurrence of words our description text.  "Clusters" of words that occur with each other with some regularity represent *topics*, though to a human such topics might be a little challenging to comprehend. Still, we can score each description based on its alignment with each of the topics discovered across the descriptions.  Those per-topic scores then provide us the basis for locating similar documents in the dataset.
# MAGIC 
# MAGIC To perform the LDA calculations, we must first standardize the words in our descriptions using lemmatization, just as we did with our title data:

# COMMAND ----------

# DBTITLE 1,Standardize Words
# declare the udf
@pandas_udf(ArrayType(StringType()))
def lemmatize_words(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    
    # setup wordnet part-of-speech info
    pos_corpus = nltk.corpus.wordnet
    pos_dict = {
      'J': nltk.corpus.wordnet.ADJ,
      'N': nltk.corpus.wordnet.NOUN,
      'V': nltk.corpus.wordnet.VERB,
      'R': nltk.corpus.wordnet.ADV
      }
    
    # initialize lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
  
    def get_lemmata(words):
      
      # determine part of speech for each word
      pos_words = nltk.pos_tag(words.tolist())
      
      # lemmatize words in relevant parts of speech
      lemmata = []
      for word, pos in pos_words:
          # just use first char of part of speech
          pos = pos[0]
          
          # if part of speech of interest, lemmatize
          if pos in pos_dict:
            lemmata += [lemmatizer.lemmatize(word, pos_dict[pos])] 
            
      return lemmata
  
    # for each set of words from the iterator
    for words in iterator:
        yield words.apply(get_lemmata)

# use function to convert words to lemmata
description_lemmata = (
  description_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'description', 'lemmata')
    ).cache()

display(description_lemmata)

# COMMAND ----------

# MAGIC %md We must now count the occurrence of the words in our dataset.  Because we are working with many more words than were found in our titles, we'll use the [HashingTF transform](https://spark.apache.org/docs/2.2.0/ml-features.html#tf-idf) to do this work.  
# MAGIC 
# MAGIC While the HashingTF and CountVectorizer appear to be performing the same work, the HashingTF transform is using a short-cut to speed up the process. Instead of creating a distinct list of words from across all the words in our descriptions, calculating term-frequency scores for each, and then limiting the resulting vector to the top occurring of those words, the HashingTF transform hashes each word to an integer value and uses that hashed value as the word's index in our vector. This allows us to skip the step of creating a word-lookup table and performing our count against it.  Instead, we can simply calculate a hash and add one to the value in the associated index position.  The trade off is that hash collisions will occur so that there will be situations where two unrelated words are counted as if they were the same.  If we are willing to accept the possibility of a little sloppiness in order to pick up substantial performance gains, then the HashingTF transform is the way to go:

# COMMAND ----------

# DBTITLE 1,Count Words
# get word counts from descriptions
description_tf = (
  HashingTF(
    inputCol='lemmata', 
    outputCol='tf',
    numFeatures=262144  # top n words to consider
    )
    .transform(description_lemmata)
  )

display(description_tf)

# COMMAND ----------

# MAGIC %md With word counts in place, we can now apply LDA to define topics and score our descriptions relative to these.  Depending on the number of iterations and the size of your cluster, this step may take a while to complete.  With that in mind, notice that we are *learning* our LDA topics using a 25% random sample of our overall dataset which should allow us to arrive at valid topics while limiting computation time:

# COMMAND ----------

# DBTITLE 1,Apply LDA
# identify LDA topics & score descriptions against these
description_lda = (
  LDA(
    k=100, 
    maxIter=20,
    featuresCol='tf',
    optimizer='online' # use the online optimizer for scalability
    )
    .fit(description_tf.sample(False, 0.25)) # train on a random sample of the data
    .transform(description_tf) # transform all the data
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_lda', ignore_errors=True)

# persist as delta table
(
  description_lda
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_lda')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_lda`').select('id','asin','description','topicDistribution')
  )

# COMMAND ----------

# MAGIC %md The LDA scores now serve as features with which we can evaluate description similarities.  As before, we must normalize these before proceeding with that step:

# COMMAND ----------

# DBTITLE 1,Normalize LDA Features
description_lda = spark.table('DELTA.`/mnt/reviews/tmp/description_lda`')

description_lda_norm = (
  Normalizer(inputCol='topicDistribution', outputCol='features', p=2.0)
    .transform(description_lda)
  ).cache()

display(description_lda_norm.select('id','asin','description', 'features'))

# COMMAND ----------

# MAGIC %md Before looking at how we will calculate similarities using our normalized features, let's examine another dimension reduction technique.

# COMMAND ----------

# MAGIC %md # Step 2b: Extract Word2Vec *Concept* Features
# MAGIC 
# MAGIC LDA scores our descriptions on their relationship to discovered topics using words found anywhere in the description. In other words, the sequencing of words in a description is not taken into consideration.  [Word2Vec](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/) on the other hand examines word proximity to get at *concepts* within the descriptions:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_w2v.gif" width="600">
# MAGIC 
# MAGIC **NOTE** Word2Vec does not require any preprocessing other than tokenization.  Also, note that Word2Vec can take a while to run so that we're fitting the model on only 25% of our data.

# COMMAND ----------

# DBTITLE 1,Apply Word2Vec
description_words = description_words.cache()

# generate w2v set
description_w2v =(
  Word2Vec(
    vectorSize=100,
    minCount=3,              # min num of word occurances required for consideration
    maxSentenceLength=1000,  # max num of words in description to consider
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
    .fit(description_words.sample(False, 0.25))
    .transform(description_words)
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_w2v', ignore_errors=True)

# persist as delta table
(
  description_w2v
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_w2v')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`').select('id','asin','description','w2v')
  )

# COMMAND ----------

# MAGIC %md As with our LDA-derived features, our Word2Vec features require normalization:

# COMMAND ----------

# DBTITLE 1,Normalize Word2Vec
description_w2v = spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`')

description_w2v_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(description_w2v)
  ).cache()

display(description_w2v_norm.select('id','asin','description', 'w2v', 'features'))

# COMMAND ----------

# MAGIC %md # Step 3: Calculate Description Similarities
# MAGIC 
# MAGIC So, how might we use either our LDA or Word2Vec features to find similar items?  Previously, we used LSH and we could do that again here.  But another technique we could use is k-means clustering.
# MAGIC 
# MAGIC Using k-means clustering we assign products to clusters based on feature similarity.  We can then use cluster assignment to limit our searches for similar products to those found within a cluster (much like we limit our similarity search to products in a shared bucket using LSH).  In Spark, we have two basic options for this: traditional k-means and bisecting k-means. Either is applicable but they will produce different results.  Regardless of which you choose, you can use traditional [elbow techniques](https://bl.ocks.org/rpgove/0060ff3b656618e9136b) to identify an optimal number of clusters though consideration of query performance against the result set is also important.  Here, we'll opt for 50 clusters as this seems to provide us a reasonable split on our data.  While we don't often speak about clustering in these terms, it's important to consider the application of clustering in this scenario as an approximate technique, much like LSH:
# MAGIC 
# MAGIC **NOTE** The clustered/bucketed data is being persisted to Delta Lake for re-use in a later notebook. In addition, we are persisting the clustering model using mlflow for similar re-use.

# COMMAND ----------

# DBTITLE 1,Assign Descriptions to Clusters
clustering_model = (
  BisectingKMeans(
    k=50,
    featuresCol='features', 
    predictionCol='bucket',
    maxIter=100,
    minDivisibleClusterSize=100000
    )
    .fit(description_w2v_norm.sample(False, 0.25))
  )

descriptions_clustered = (
  clustering_model
    .transform(description_w2v_norm)
  )

# persist the clustering model for next notebook
with mlflow.start_run():
  mlflow.spark.log_model(
    clustering_model, 
    'model',
    registered_model_name='description_clust'
    )
  
# persist this data for the next notebook
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_sim', ignore_errors=True)
(
  descriptions_clustered
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/reviews/tmp/description_sim')
  )

display(
  descriptions_clustered
    .groupBy('bucket')
    .agg(count('*').alias('descriptions'))
    .orderBy('bucket')
  )

# COMMAND ----------

# MAGIC %md With our data assigned to a cluster, locating similar products is fairly straightforward.  All we need to do is perform an exhaustive comparison within each cluster/bucket.
# MAGIC 
# MAGIC When using LSH, this work including the calculation of the Euclidean distance between vectors was performed for us.  Here, we'll need to perform our distance calculation using a custom function. 
# MAGIC 
# MAGIC Euclidean distance calculations between two vectors are fairly easy to perform.  However, the pandas UDFs we used before do not have a means to accept a vector in its native format.  But Scala does.  By registering our Scala function for the calculation of Euclidean distance between two vectors with the Spark SQL engine, we can easily apply our function to data in a Spark DataFrame using Python as will be demonstrated in a later cell:

# COMMAND ----------

# DBTITLE 1,Define Function for Distance Calculations
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

# MAGIC %md And now let's make the recommendations, limiting our comparisons to those items in the same bucket (cluster) as our sample product:
# MAGIC 
# MAGIC **NOTE** We will use the same sample product as was used in the last notebook.

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Product
sample_product = descriptions_clustered.filter('asin==\'B01G6M7CLK\'')

display(sample_product)

# COMMAND ----------

# DBTITLE 1,Retrieve Similar Descriptions
display(
  descriptions_clustered
    .withColumnRenamed('features', 'features_b')
    .join(sample_product.withColumnRenamed('features', 'features_a'), on='bucket', how='inner')  # join on bucket/cluster
    .withColumn('distance', expr('euclidean_distance(features_a, features_b)')) # calculate distance
    .withColumn('raw_sim', expr('1/(1+distance)')) # convert distance to similarity score
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    .orderBy('distance', ascending=True)
    .limit(100) # get top 100 recommendations
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md Unlike our TF-IDF scored titles, the basis for matching descriptions is a little harder to intuit from a simple viewing of the data.  The notion of *topics* or *concepts* are a bit more elusive than simple word-count derived scores.  Still, a review of the descriptions gives us a sense as to why some descriptions are considered more similar than others. To constrain things a bit further, we might consider limiting our LDA and Word2Vec feature generation to a smaller number of words from the beginning of the description as this is the part of the text most likely to directly tie into the key aspects of the product.  With Word2Vec, this is done through a simple argument setting.  With LDA, we would need to add a step to truncate our tokenized words before performing lemmatization.
