# Databricks notebook source
# MAGIC %md # Introduction
# MAGIC 
# MAGIC The purpose of this notebook is to examine how features may be extracted from product titles in order to calculate similarities between products. (Other metadata fields are explored in the notebooks that follow this one.) These similarities will be used as the basis for making **Related products** recommendations:
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
import nltk

import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, lit, count, countDistinct, col, array_join, expr, monotonically_increasing_id, explode

from typing import Iterator

from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vector, Vectors

from pyspark.sql import DataFrame

import mlflow.spark

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: Prepare Title Data
# MAGIC 
# MAGIC If our goal is to recommend highly similar products, we might identify such products based on product names.  This information is captured in the *title* field within this dataset:

# COMMAND ----------

# DBTITLE 1,Retrieve Titles
# retrieve titles for each product
titles = (
  spark
    .table('reviews.metadata')
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

num_of_titles = titles.count()
num_of_titles

# COMMAND ----------

# present titles
display(
  titles
  )

# COMMAND ----------

# MAGIC %md In order to enable similarity comparisons between titles, we might employ a fairly straightforward word-based comparison where each word is weighted relative to its occurrence in the title and it's overall occurrence across all titles.  These weights are often calculated using *term-frequency - inverse document frequency* (*TF-IDF*) scores.
# MAGIC 
# MAGIC As a first step in calculating TF-IDF scores, we need to split out the words in our titles and move them to a consistent case.  This is done here using the [RegexTokenizer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RegexTokenizer):
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_coasters.jpg' width='150'>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_tokenization2.png' width='1100'>
# MAGIC 
# MAGIC Spark also makes available a simple [Tokenizer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Tokenizer) that splits text on white-space but with messy text data (such as is found in our *title* field), the RegexTokenizer allows us to better deal with things like stray punctuation characters:

# COMMAND ----------

# DBTITLE 1,Retrieve Words from Titles
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='title', 
    outputCol='words'
    )

title_words = tokenizer.transform(titles)

display(title_words)

# COMMAND ----------

# MAGIC %md With our words split out, we can now consider how to deal with common word variations such as singular vs. plural forms, future, present and past tense verbs, and other word-form variations which may cause very similar words to be seen separately.
# MAGIC 
# MAGIC One technique for this is known as [stemming](https://en.wikipedia.org/wiki/Stemming).  With stemming, common word suffixes are dropped in order to truncate a word to its root (stem).  While effective, stemming lacks knowledge about how a word is used and how words with non-standard forms, *i.e. man vs. men*, might be related. Using a slightly more sophisticated technique known as [lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) we can better standardize word forms:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_coasters.jpg' width='150'><img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_lemmatization2.png' width='1100'>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC There are a variety of libraries we can use to convert words to *lemmata* (plural of *lemma*).  [NLTK](https://buildmedia.readthedocs.org/media/pdf/nltk/latest/nltk.pdf) is one of the more popular of these and is pre-installed in most Databricks ML runtime environments.  Still, to perform lemmatization with NLTK, we need to install a tagged corpus and part of speech (POS) predictor on each worker node in our cluster.  This is done by configuring our cluster with the init script as explained at the top of this notebook.
# MAGIC 
# MAGIC Here, we are using the [WordNet corpus](https://www.nltk.org/howto/wordnet.html) to provide context for our words.  We'll use this context to not only standardize words but to eliminate words that are not commonly used as adjectives, nouns, verbs or adverbs, the parts of speech that typically carry the most information.  Alternative *corpora* (plural of *corpus*) may be [downloaded](http://www.nltk.org/howto/corpus.html) with NLTK and may give you different results:
# MAGIC 
# MAGIC **NOTE** I'm implementing the lemmatization logic using a pandas UDF with an *iterator of series to iterator of series* type.  This is a [new style of pandas UDF (in Spark 3.0)](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html) which is useful in scenarios where expensive initialization takes place.

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
title_lemmata = (
  title_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'title', 'lemmata')
    ).cache()

display(title_lemmata)

# COMMAND ----------

# MAGIC %md # Step 2: Calculate TF-IDF Scores
# MAGIC 
# MAGIC With our data prepared, we can now proceed with the calculation of the *term-frequency* portion of our TF-IDF calculation.  Here, we may take a simple count of the occurrence of words within a title.  Because titles are typically succinct, we would expect that most words will occur only once in a given title.  To avoid counting more rare words that will not help us with a similarity comparison, we will limit the number of words we will count to the top 262,144 words across all our titles.  This is the default configuration of the word counters in Spark but we're explicitly assigning this value in the code to make it clear that a limit is in place.
# MAGIC 
# MAGIC To count words, we have two basic options.  The [CountVectorizer](https://spark.apache.org/docs/latest/ml-features.html#countvectorizer) performs the count through a brute-force exercise which works fine for smaller text content.  We will make use of the alternative term-frequency transformer, HashTF, in the next notebook: 

# COMMAND ----------

# DBTITLE 1,Count Word Occurrences in Titles
# count word occurences
title_tf = (
  CountVectorizer(
    inputCol='lemmata', 
    outputCol='tf',
    vocabSize=262144  # top n words to consider
    )
    .fit(title_lemmata)
    .transform(title_lemmata)
    )

display(title_tf.select('id','asin','lemmata','tf'))

# COMMAND ----------

# MAGIC %md Now we can calculate the *inverse document frequency* (IDF) for the words in our titles. As a word is used more and more frequently across titles, it's IDF score will decrease logarithmically indicating it carries less and less differentiating information.  The raw IDF scores are typically multiplied against the TF scores to produce the desired TF-IDF score:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/review_tfidf.png" width="400">
# MAGIC 
# MAGIC All of is tackled through the [IDF transform](https://spark.apache.org/docs/latest/ml-features.html#tf-idf):

# COMMAND ----------

# DBTITLE 1,Calculate TF-IDF Scores
# calculate tf-idf scores
title_tfidf = (
  IDF(inputCol='tf', outputCol='tfidf')
    .fit(title_tf)
    .transform(title_tf)
  )

display(
  title_tfidf.select('id','asin','lemmata','tfidf')
  )

# COMMAND ----------

# MAGIC %md The TF-IDF scores returned by the transform are not normalized.  With similarity calculations based on the distance calculations, we frequently apply an L2-normalization.  This is addressed by applying the [Normalizer](https://spark.apache.org/docs/latest/ml-features.html#normalizer) transform to our TF-IDF scores:

# COMMAND ----------

# DBTITLE 1,Normalize the TF-IDF Values
# apply normalizer
title_tfidf_norm = (
  Normalizer(inputCol='tfidf', outputCol='tfidf_norm', p=2.0)
    .transform(title_tfidf)
  )

display(title_tfidf_norm.select('id','asin','lemmata','tfidf','tfidf_norm'))

# COMMAND ----------

# MAGIC %md # Step 3: Identify Products with Similar Titles
# MAGIC 
# MAGIC We now have features with which we can calculate similarities between titles. The brute-force approach would have us compare each of our 11.8 million titles to one another, resulting in about 70-trillion comparisons.  This is not economically viable, even in a distributed system.  Instead, we'll need to find a short cut that allows us to limit our comparisons to those products most likely to be similar to one another.
# MAGIC 
# MAGIC In the collaborative-filtering notebooks, we examined [Local Sensitive Hashing](https://spark.apache.org/docs/latest/ml-features.html#locality-sensitive-hashing) as one approach to tackling this problem.  (For a deeper dive on LSH, please refer to those notebooks.) We can apply that same technique here to find similar titles:

# COMMAND ----------

# DBTITLE 1,Apply LSH to Titles
# configure lsh
bucket_length = 0.0001
lsh_tables = 5

# fit the algorithm to the dataset 
fitted_lsh = (
  BucketedRandomProjectionLSH(
    inputCol = 'tfidf_norm', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    ).fit(title_tfidf_norm)
  )

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(title_tfidf_norm)
    ).cache()

display(
  hashed_vectors.select('id','asin','title','tfidf_norm','hash')
  )

# COMMAND ----------

# MAGIC %md Using LSH, we've been able to sort titles into buckets of *approximately* similar values very quickly.  The technique is not perfect, but as we can see with a sample product, we can locate similar products reasonably well:

# COMMAND ----------

# DBTITLE 1,Extract Information for Sample Product
# retrieve data for example product
sample_product = hashed_vectors.filter('asin==\'B01G6M7CLK\'') 
                                       
display(
  sample_product.select('id','asin','title','tfidf_norm','hash')
  )                                     

# COMMAND ----------

# DBTITLE 1,Retrieve 100 Most Similar Products 
number_of_titles = 100

# retrieve n nearest customers 
similar_k_titles = (
  fitted_lsh.approxNearestNeighbors(
    hashed_vectors, 
    sample_product.collect()[0]['tfidf_norm'], 
    number_of_titles, 
    distCol='distance'
    )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select('id', 'asin', 'title', 'distance', 'similarity')
  )
  
display(similar_k_titles)

# COMMAND ----------

# MAGIC %md From a quick review of the data, we can see we're finding quite a few items with similar titles.  We might examine the breadth of products within the category within which our sample product resides to get a better sense of how complete our approach is, but no matter what, there's a degree of subjectivity in our evaluation that is difficult for us to avoid. This is a very common challenge in exploring recommendations where there is no baseline truth against which to compare (such as a product rating or a purchase event).  All we can do here is play with settings to arrive at what appears to be a reasonably good set of recommendations and then perform a limited test with real customers to see how they respond to our suggestions.

# COMMAND ----------

# DBTITLE 1,Drop Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
