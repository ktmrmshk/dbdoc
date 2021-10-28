# Databricks notebook source
# MAGIC %md The purpose of this notebook is to explore how the content-based recommenders developed in previous notebooks might be operationalized. This notebook should be run on a **Databricks ML 7.3+ cluster**.

# COMMAND ----------

# MAGIC %md # Introduction
# MAGIC 
# MAGIC No recommendation solution should be constructed with out careful consideration of the data volumes, the frequency of data changes, the service level expectations for the recommendations and the business goals the recommendations are intended to drive. That said, we might anticipate some design choices used by many recommenders and build a demonstration deployment archtiecture to serve as a starting point for these considerations:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_feature_pipelines5.png" width="700">
# MAGIC 
# MAGIC To enable this particular architecture, we'll need to bring together the feature engineering logic demonstrated in previous notebooks as a manageable unit we might refer to as a pipeline.  Raw information will flow into the pipeline and transformed features will flow out.  We'll implement three pipelines for each of the feature sets related to titles, descriptions and categories, again, as shown in prior notebooks. We will keep the feature output from each of these pipelines stored in separate tables as such a pattern will make it easier for us to schedule and roll out changes to the pipelines separately. Similarly, we'll need to construct a pipeline for our user-profiles.  It's important that we recognize that this pipeline will consume both user feedback (in the form of ratings) as well as features derived through our other pipelines. 
# MAGIC 
# MAGIC Feature generation is the first challenge to be addressed by our recommender solution. The next challenge is the generation of the recommendations themselves.  One approach is to pre-compute the *related products* recommendations for some or all of our products. Another might be to dynamically generate these recommendations and then cache them for reuse. Here, we'll focus on the pre-computation pattern.

# COMMAND ----------

# MAGIC %md **NOTE** This cluster used to run this notebook should be created using a [cluster-scoped initialization script](https://docs.databricks.com/clusters/init-scripts.html?_ga=2.158476346.1681231596.1602511918-995336416.1592410145#cluster-scoped-init-script-locations) which installs the NLTK WordNet corpus and Averaged Perceptron Tagger.  The following cell can be used to generate such a script but you must associate it with the cluster before running any code that depends upon it:

# COMMAND ----------

# DBTITLE 1,Generate Cluster Init Script
dbutils.fs.mkdirs('dbfs:/databricks/scripts/')

dbutils.fs.put(
  '/databricks/scripts/install-nltk-downloads.sh',
  '''
  #!/bin/bash\n/databricks/python/bin/python -m nltk.downloader wordnet\n/databricks/python/bin/python -m nltk.downloader averaged_perceptron_tagger''', 
  True
  )

# alternatively, you could install all NLTK elements with: python -m nltk.downloader all

# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/install-nltk-downloads.sh')
  )

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark import keyword_only
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH, SQLTransformer, HashingTF, Word2Vec
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.stat import Summarizer

import nltk

import mlflow.spark

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, expr, col, lit, struct, collect_list, count
from typing import Iterator
from pyspark.sql.types import *

from delta.tables import *

import pandas as pd

import shutil
import os

# COMMAND ----------

# MAGIC %md # Step 1: Assemble Product Feature Pipelines
# MAGIC 
# MAGIC Our content recommender uses three feature sets derived from the *title*, *description* & *category* fields in our product metadata. Each of these feature sets is created through a series of various transformations which must be applied in a specific sequence. To preserve information about these transformations, we can organize them into pipelines which we can then fit and persist for re-use between cycles. The first of these that we will build is the pipeline for our title transformations.
# MAGIC 
# MAGIC The basic transformation steps we will perform against our title data are as follows:</p>
# MAGIC 1. Tokenize the title to produce a bag of words
# MAGIC 2. Lemmatize the bag of words
# MAGIC 3. Calculate token frequency
# MAGIC 4. Calculate TF-IDF scores
# MAGIC 5. Normalize the TF-IDF scores
# MAGIC 
# MAGIC All but one of these steps is addressed through out-of-the-box transformers.  For the lemmatization step, we'll need to write a custom transformer.  The steps for this are nicely presented in [this post](https://stackoverflow.com/questions/32331848/create-a-custom-transformer-in-pyspark-ml) and implemented in this next cell.  Notice that the *_transform* method of the custom transformer class is a copy-and-paste of the UDF  created in prior notebooks:

# COMMAND ----------

# DBTITLE 1,Define Custom Transform for Lemmatization
class NLTKWordNetLemmatizer(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(NLTKWordNetLemmatizer, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, lemmatize_words(in_col))

# COMMAND ----------

# MAGIC %md With our custom transformer in place, we can now construct our titles pipeline:

# COMMAND ----------

# DBTITLE 1,Assemble Titles Pipeline
# step 1: tokenize the title
title_tokenizer = RegexTokenizer(
  minTokenLength=2, 
  pattern='[^\w\d]', 
  inputCol='title', 
  outputCol='words'
  )

# step 2: lemmatize the word tokens
title_lemmatizer = NLTKWordNetLemmatizer(
  inputCol='words',
  outputCol='lemmata'
  )

# step 3: calculate term-frequencies
title_tf = CountVectorizer(
  inputCol='lemmata', 
  outputCol='tf',
  vocabSize=262144
  )
  
# step 4: calculate inverse document frequencies
title_tfidf = IDF(
  inputCol='tf', 
  outputCol='tfidf'
  )
  
# step 5: normalize tf-idf scores
title_normalizer = Normalizer(
  inputCol='tfidf', 
  outputCol='tfidf_norm', 
  p=2.0
  )

# step 6: assign titles to buckets
title_lsh = BucketedRandomProjectionLSH(
  inputCol = 'tfidf_norm', 
  outputCol = 'hash', 
  numHashTables = 5, 
  bucketLength = 0.0001
  )

# assemble pipeline
title_pipeline = Pipeline(stages=[
  title_tokenizer,
  title_lemmatizer,
  title_tf,
  title_tfidf,
  title_normalizer,
  title_lsh
  ])

# COMMAND ----------

# MAGIC %md We can now fit the pipeline and perform a transformation just to verify it works as expected.  Please note that fitting the pipeline is the only step actually required before persisting the pipeline in later cells:

# COMMAND ----------

# DBTITLE 1,Fit & Test Titles Pipeline
# retrieve titles
titles = (
  spark
    .table('reviews.metadata')
    .repartition(sc.defaultParallelism)
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

# fit pipeline to titles
title_fitted_pipeline = title_pipeline.fit(titles)

# present transformed data for validation
display(
  title_fitted_pipeline.transform(titles.limit(1000))
  )

# COMMAND ----------

# MAGIC %md Let's now turn our attention to pipeline persistence.  To persist the fitted pipeline, we'll make use of the [mlflow registry](https://mlflow.org/docs/1.4.0/model-registry.html):
# MAGIC 
# MAGIC **NOTE** The mlflow registry is designed to enable an MLOps workflow where models are tested and moved between stages over time.  If you immediately attempt to retrieve the pipeline after it is logged, you may receive an error. If you are troubleshooting persistence and retrieval, be sure to give the registry a few seconds between steps to ensure the pipeline object is properly available, otherwise, you might receive an error that the object cannot be found.

# COMMAND ----------

# DBTITLE 1,Persist Fitted Titles Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    title_fitted_pipeline, 
    'pipeline',
    registered_model_name='title_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md Now we can create a pipeline for our description data.  We'll use the Word2Vec functionality explored in prior notebooks.  Our pipeline will consist of the following steps:</p>
# MAGIC 
# MAGIC 1. Flatten description from array to string
# MAGIC 2. Tokenize the description
# MAGIC 2. Apply Word2Vec transformation
# MAGIC 3. Normalize Word2Vec scores
# MAGIC 4. Calculate KMeans clusters
# MAGIC 
# MAGIC As there are no custom transformers required with this pipeline, but we will need to introduce a [SQL transformer](https://spark.apache.org/docs/latest/ml-features#sqltransformer) to handle our first step which was previously addressed in a query:
# MAGIC 
# MAGIC **NOTE** We are limiting the words considered in Word2Vec to the first 200 per information discussed in the prior, related notebook. Also, it's important to point out that the SQL transformer has access to all incoming fields in the dataset but will only pass along those specified in its SELECT-list.

# COMMAND ----------

# DBTITLE 1,Assemble Descriptions Pipeline
# step 1: flatten the description to form a single string
descript_flattener = SQLTransformer(
    statement = '''
    SELECT id, array_join(description, ' ') as description 
    FROM __THIS__ 
    WHERE size(description)>0'''
    )

# step 2: split description into tokens
descript_tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]', 
    inputCol='description', 
    outputCol='words'
    )

# step 3: convert tokens into concepts
descript_w2v =  Word2Vec(
    vectorSize=100,
    minCount=3,              
    maxSentenceLength=200,  
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
  
# step 4: normalize concept scores
descript_normalizer = Normalizer(
    inputCol='w2v', 
    outputCol='features', 
    p=2.0
    )

# step 5: assign titles to buckets
descript_cluster = BisectingKMeans(
    k=50,
    featuresCol='features', 
    predictionCol='bucket',
    maxIter=100,
    minDivisibleClusterSize=100000
    )

# assemble the pipeline
descript_pipeline = Pipeline(stages=[
    descript_flattener,
    descript_tokenizer,
    descript_w2v,
    descript_normalizer,
    descript_cluster
    ])

# COMMAND ----------

# DBTITLE 1,Fit & Test Descriptions Pipeline
# retrieve descriptions
descriptions = (
    spark
      .table('reviews.metadata')
      .select('id','description')
      .repartition(sc.defaultParallelism)
      .sample(False, 0.25)
    )

# fit pipeline
descript_fitted_pipeline = descript_pipeline.fit(descriptions)

# verify transformation
display(
  descript_fitted_pipeline.transform(descriptions.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,Persist Fitted Descriptions Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    descript_fitted_pipeline, 
    'pipeline',
    registered_model_name='descript_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md And now we tackle the categories pipeline.  This pipeline will perform the following steps:</p>
# MAGIC 
# MAGIC 1. Alter category levels to include lineage
# MAGIC 2. Perform one-hot encoding of category levels
# MAGIC 3. Identify the root member
# MAGIC 
# MAGIC The first of these steps is handled through a custom transformer. The last of these steps is handled through a SQL transformer:

# COMMAND ----------

# DBTITLE 1,Define Custom Transformer
# define custom transformer to flatten the category names to include lineage
class CategoryFlattener(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(CategoryFlattener, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, cleanse_category(in_col))

# COMMAND ----------

# DBTITLE 1,Assemble Categories Pipeline
# step 1: flatten hierarchy
category_flattener = CategoryFlattener(
    inputCol='category',
    outputCol='category_clean'
    )

# step 2: one-hot encode hierarchy values
category_onehot = HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )

# step 3: assign bucket
category_cluster = SQLTransformer(
    statement='SELECT id, category[0] as bucket, category, category_clean, category_onehot FROM __THIS__ WHERE size(category)>0'
    )

# assemble pipeline
category_pipeline = Pipeline(stages=[
    category_flattener,
    category_onehot,
    category_cluster
    ])

# COMMAND ----------

# DBTITLE 1,Fit & Test Categories Pipeline
# retrieve categories
categories = (
    spark
      .table('reviews.metadata')
      .select('id','category')
      .repartition(sc.defaultParallelism)
    )

# fit pipeline
category_fitted_pipeline = category_pipeline.fit(categories)

# test the transformation
display(
  category_fitted_pipeline.transform(categories.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,Persist Fitted Categories Pipeline
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    category_fitted_pipeline, 
    'pipeline',
    registered_model_name='category_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md # Step 2: Generate Product Features
# MAGIC 
# MAGIC With our pipelines defined, trained & persisted, we can now focus attention on generating the features from which we will build our recommendations.  When we roll out a new pipeline, we'll need to initialize our features tables with the new information.  Once this is done, we will need to rerun our pipeline as new products are added to our metadata table or product metadata changes.
# MAGIC 
# MAGIC Ideally, we'll have in our product metadata table a *modified date time* value with which we can reliably detect these changes.  We don't have such a field in our metadata table so we can't implement this logic.  That said, we will show how the tables for each feature set can be initialized and updated for those who have the ability to perform change detection.

# COMMAND ----------

# MAGIC %md The first step in executing our pipelines is to retrieve the pipeline objects from mlflow.  Earlier, we saved our pipeline objects to the mlflow registry.  The registry will allow us to move pipeline objects through *Staging*, *Production* and *Archived* stages and simplify the retrieval of the current *Production* version of the pipeline object for this process.  That said, we did not move our pipelines programmatically into a *Production* stage (as this would typically be performed through a separate process involving formal evaluation), and for that reason, we'll grab the last version of each pipeline currently residing in the *None* stage. (Again, this is **not** how this would typically be done in a production scenario.)
# MAGIC 
# MAGIC Let's see how this works for our titles pipeline:
# MAGIC 
# MAGIC **NOTE** This next cell will generate an error.

# COMMAND ----------

# DBTITLE 1,Attempt to Retrieve Titles Pipeline
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'  # models:/<model name defined at persistence>/<stage>
    )

# COMMAND ----------

# MAGIC %md There is a large verbose warning that is generated which can be ignored.  What we want to focus our attention on is the **AttributeError** that states that the class for our custom transformer cannot be found, *i.e.* *module '__main__' has no attribute 'NLTKWordNetLemmatizer'*.  The inability of the pipeline to find the class definition (even though it resides in this same notebook) is sometimes referred to as the [*leaky pipeline* problem](https://rebeccabilbro.github.io/module-main-has-no-attribute/) and its commonly encountered with *sklearn* which Spark ML closely emulates.
# MAGIC 
# MAGIC To overcome this problem, we have two options: </p>
# MAGIC 
# MAGIC 1. Include the class definition with any code that retrieves the pipeline object and force the top-level environment to recognize it OR
# MAGIC 2. Create a *whl* file which we can attach to the cluster
# MAGIC 
# MAGIC The creation of a *whl* file is covered [here](https://packaging.python.org/guides/distributing-packages-using-setuptools/) (and [here](https://medium.com/swlh/beginners-guide-to-create-python-wheel-7d45f8350a94)) and the techniques for attaching such a file to a Databricks cluster are addressed [here](https://docs.databricks.com/libraries/index.html). This is the more elegant solution but one which requires capabilities which are not easily demonstrated within the confines of a notebook. For that reason, we'll solve the problem with the inelegant first solution:

# COMMAND ----------

# DBTITLE 1,Force Environment to See Custom Transformer in This Notebook
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'NLTKWordNetLemmatizer', NLTKWordNetLemmatizer)

# COMMAND ----------

# DBTITLE 1,Retrieve Titles Pipeline
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'
    )

# COMMAND ----------

# MAGIC %md Let's go ahead and retrieve the remaining pipelines:

# COMMAND ----------

# DBTITLE 1,Retrieve Descriptions Pipeline
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# COMMAND ----------

# DBTITLE 1,Retrieve Categories Pipeline
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'CategoryFlattener', CategoryFlattener)

# retrieve pipeline from registry
retrieved_category_pipeline = mlflow.spark.load_model(
    model_uri='models:/category_fitted_pipeline/None'
    )

# COMMAND ----------

# MAGIC %md Now, let's generate features for our products.  To get started, we will create an empty dataframe with a schema as would be expected to be created by our pipeline. We will append this to our target destination in order to *lay down* a DeltaLake table with a schema that will allow us to perform a merge operation later.  We have to do this instead of using a *CREATE TABLE IF NOT EXISTS* statement because the vector user-defined type is not recognized in such a statement.  Creating the DeltaLake table in this manner ensures that we have a target schema for the merge but that we do not modify any data if such a schema is already in place:

# COMMAND ----------

# DBTITLE 1,Create Title Features DeltaLake Object (If Needed)
# retreive an empty dataframe
titles_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','title')
  )

# transform empty dataframe to get transformed structure
title_dummy_features = (
  retrieved_title_pipeline
    .transform(titles_dummy)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
title_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/title_pipeline_features')
  )

# COMMAND ----------

# MAGIC %md Now we can move our real data updates into the schema.  Notice where we would have applied some change detection logic in determining which titles to process as part of this pipeline effort:

# COMMAND ----------

# DBTITLE 1,Transform Titles to Features
# retrieve titles to process for features
titles_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','title')
  )

# generate features
title_features = (
  retrieved_title_pipeline
    .transform(titles_to_process)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# COMMAND ----------

# DBTITLE 1,Merge Title Features
# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/title_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      title_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'hash':'source.hash'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md We can now apply this pattern to our description & category features:

# COMMAND ----------

# DBTITLE 1,Generate Description Features
# retreive an empty dataframe
descript_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','description')
  )

# transform empty dataframe to get transformed structure
descript_dummy_features = (
  retrieved_descript_pipeline
    .transform(descript_dummy)
    .selectExpr('id','w2v','features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
descript_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/descript_pipeline_features')
  )

# retrieve titles to process for features
descriptions_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','description')
  )

# generate features
description_features = (
  retrieved_descript_pipeline
    .transform(descriptions_to_process)
    .selectExpr('id','w2v','features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/descript_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      description_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'w2v':'source.w2v', 'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
  )

# COMMAND ----------

# DBTITLE 1,Examine Distribution by Bucket
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as products
# MAGIC FROM DELTA.`/mnt/reviews/gold/descript_pipeline_features`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# DBTITLE 1,Generate Category Features
# retreive an empty dataframe
category_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','category')
  )

# transform empty dataframe to get transformed structure
category_dummy_features = (
  retrieved_category_pipeline
    .transform(category_dummy)
    .selectExpr('id','category_onehot as features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
category_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/category_pipeline_features')
  )

# retrieve categories to process for features
category_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','category')
  )

# generate features
category_features = (
  retrieved_category_pipeline
    .transform(category_to_process)
    .selectExpr('id','category_onehot as features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/category_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      category_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md Returning to our *strawman* architecture, we've implemented the following components:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_feature_pipelines7.png" width="700">
# MAGIC 
# MAGIC Now let's turn to user-profiles to close out the feature engineering side of the solution.

# COMMAND ----------

# MAGIC %md # Step 3: Generate User-Profiles
# MAGIC 
# MAGIC To generate user-profiles, we won't assemble a pipeline. This is because the work is much more easily implemented using DataFrame manipulations. But like the earlier pipelines, we might want to consider using some kind of change detection to limit the number of profiles we need to generate on a given cycle.  
# MAGIC 
# MAGIC With our review data, we do have access to a *unixReviewTime* field with which we could identify new reviews. Depending on how we approach the problem of user-profile generation, we might consider creating profiles for just those users with new reviews. The code for detecting new reviews based on the current date is included here for review but commented out so that we are generating profiles based on all available reviews (that meet the criteria discussed in the last notebook):

# COMMAND ----------

# DBTITLE 1,Generate Raw User-Profile
user_profiles_raw = (
  spark
    .sql('''
      SELECT
        a.reviewerID,
        a.overall as rating,
        b.w2v
      FROM reviews.reviews a
      INNER JOIN DELTA.`/mnt/reviews/gold/descript_pipeline_features` b
        ON a.product_id=b.id
      --INNER JOIN (  -- logic to identify reviewers with new relevant reviews in last 1 day
      --  SELECT DISTINCT
      --    reviewerID
      --  FROM reviews.reviews
      --  WHERE 
      --      unixReviewTime >= unix_timestamp( date_sub(current_date(), 1) ) AND
      --      overall >= 4 AND
      --      product_id Is Not Null
      --  ) c
      --  ON a.reviewerID=c.reviewerID
      WHERE a.overall >= 4 AND a.product_id Is Not Null
    ''')
  .groupBy('reviewerID')
    .agg(
      Summarizer.mean(col('w2v'), weightCol=col('rating')).alias('w2v') 
      )
  )

# COMMAND ----------

# MAGIC %md It's important to note that the Normalizer transform has no dependency on training. As such, we can create one as needed, instead of being concerned about persisting one for re-use:

# COMMAND ----------

# DBTITLE 1,Normalize Profile Scores
user_profiles_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(user_profiles_raw)
   )

# COMMAND ----------

# MAGIC %md Unlike the Normalizer, the clustering model used to assign our profiles to buckets does depend on prior training. It's important that the model we retrieve is associated with the description features accessed in the first step of our user-profile creation.  Here, we'll access that model directly from the pipeline, by-passing the transformations leading into it:

# COMMAND ----------

# DBTITLE 1,Assign Profiles to Bucket
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# assign profiles to clusters/buckets
user_profiles = (
  retrieved_descript_pipeline.stages[-1].transform(user_profiles_norm)
  )

# COMMAND ----------

# MAGIC %md We can now persist our user-profiles and their bucket/cluster assignments:

# COMMAND ----------

# DBTITLE 1,Persist Profiles 
# retreive an empty dataframe
user_profiles_dummy_features = (
  user_profiles
    .filter('1==2')
    .select('reviewerID','features','bucket')
  )

# transform empty dataframe to get transformed structure
(
user_profiles_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/user_profile_pipeline_features')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/user_profile_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      user_profiles.alias('source'),
      condition='target.reviewerID=source.reviewerID'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md The *Featuring Engineering* portion of our architecture has now been completed.

# COMMAND ----------

# MAGIC %md # Step 4a: Create Product-Based Recommendations
# MAGIC 
# MAGIC With features in place and being kept up-to-date with jobs leveraging the pipelines presented in the last few steps, we now can consider how we might publish our recommendations to the application layer. Performance with regards to the retrieval of the recommendations is going to be essential, but with so many products in the portfolio, generating and caching recommendations ahead of time may be time-consuming and cost-prohibitive. 
# MAGIC 
# MAGIC The right solution will strike a balance between all these concerns, and as was mentioned at the top of this notebook, depends on the input of numerous parties, but we imagine some amount of caching, whether its populated through batch processes are on-the-fly to enable some measure of re-use will be a component of many solutions. Popular targets for the caching and serving of recommendation data include the following, all of which can be written to by Databricks:</p>
# MAGIC 
# MAGIC * [Redis](https://docs.databricks.com/data/data-sources/redis.html)
# MAGIC * [MongoDB](https://docs.databricks.com/data/data-sources/mongodb.html)
# MAGIC * [CouchBase](https://docs.databricks.com/data/data-sources/couchbase.html)
# MAGIC * [Azure CosmosDB](https://docs.databricks.com/data/data-sources/azure/cosmosdb-connector.html)
# MAGIC * AWS DocumentDB: See info regarding MongoDB or consider using [AWS Glue](https://aws.amazon.com/about-aws/whats-new/2020/04/aws-glue-now-supports-reading-from-amazon-documentdb-and-mongodb-tables/) for orchestration
# MAGIC 
# MAGIC For this solution, we will focus on the logic required to pre-generate recommendations and simply display the results.  Code samples provided in the links above should enable you to move the assembled recommendations from the DataFrames into the targeted caching platform. 

# COMMAND ----------

# MAGIC %md Our first step is to determine for which products we will be making recommendations.  We'll limit our processing to 10,000 products in the *Home & Kitchen* category.  In the real world, we might need to process more or fewer products in order to work our way efficiently through product backlogs or to keep up with changes in our product portfolio.
# MAGIC 
# MAGIC We'll refer to the products for which we are making recommendations as our **a-product**s and the products that make up potential recommendations as our **b-product**s.  The b-products are limited here to products within the same high-level product category, *i.e.* *Home & Kitchen*, which, again, may or may not be the right approach for your needs:

# COMMAND ----------

# DBTITLE 1,Determine Products with which to Make Recommendations
## parallelize downstream work
#spark.conf.set('spark.sql.shuffle.partitions', sc.defaultParallelism * 10)

# products we ae making recommendations for
product_a =  (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_features`')
    .filter("bucket = 'Home & Kitchen'")
    .limit(10000) # we would typically have some logic here to identify products that we need to generate recommendations for
    .selectExpr('id as id_a')
  ).cache()

# recommendation candidates to consider
product_b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_features`')
    .filter("bucket = 'Home & Kitchen'")
    .selectExpr('id as id_b')
  ).cache()

# COMMAND ----------

# MAGIC %md Let's now calculate similarities using our content-derived features.  We envision this process as separate from the feature generation steps above, but we will leverage the already retrieved pipelines in the code below to avoid repeating that step.
# MAGIC 
# MAGIC Notice that we are setting some limits on similarity scores and on distance between products (in the LSH lookup).  These limits are implemented to reduce the volume of data headed into the final recommendation scoring calculation.  Knowledge of your data and your business needs should be used to inform how or if such restrictions should be implemented:

# COMMAND ----------

# DBTITLE 1,Define Similarity Functions
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for jaccard similarity calc
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,Calculate Category Similarities
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .withColumn('features_a', col('features'))
    .withColumn('id_a',col('id'))
    .drop('features','id')
    )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .withColumn('features_b', col('features'))
    .withColumn('id_b',col('id'))
    .drop('features','id')
    )

# similarity results
category_similarity = (
  a.crossJoin(b)
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .select('id_a', 'id_b', 'similarity')
    )

display(category_similarity)

# COMMAND ----------

# DBTITLE 1,Calculate Title Similarities
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .selectExpr('id as id_a','features as tfidf_norm')
  )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .selectExpr('id as id_b','features as tfidf_norm')
  )

# retrieve fitted LSH model from pipeline
fitted_lsh = retrieved_title_pipeline.stages[-1]

# similarity results
title_similarity = (
  fitted_lsh.approxSimilarityJoin(
      a,  
      b,
      threshold = 1.4, # this is pretty high so that we can be more inclusive.  Setting it at or above sqrt(2) will bring back every product
      distCol='distance'
      )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .selectExpr('datasetA.id_a as id_a', 'datasetB.id_b as id_b', 'similarity')
    .filter('similarity > 0.01') # set a lower limit for consideration
    )

display(title_similarity)

# COMMAND ----------

# MAGIC %md **NOTE** We've included the code for calculating description-based similarities though we are not using this in the final calculations at the end of this notebook.

# COMMAND ----------

# DBTITLE 1,Calculate Description Similarities
## categories for products for which we wish to make recommendations
#a = (
#  spark
#    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
#    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
#    .selectExpr('id as id_a', 'features as features_a', 'bucket')
#  )
#
## categories for products which will be considered for recommendations
#b = (
#  spark
#    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
#    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
#    .selectExpr('id as id_b', 'features as features_b', 'bucket')
#  )
#
## caculate similarities
#description_similarity = (
#  a  
#    .hint('skew','bucket')
#    .join(b, on=[a.bucket==b.bucket], how='inner')
#    .withColumn('similarity', expr('euclidean_similarity(features_a, features_b)'))
#    .select('id_a','id_b','similarity')
#    .filter('similarity > 0.01') # set a lower limit for consideration
#  )
#
#display(description_similarity)

# COMMAND ----------

# MAGIC %md With our similarity scores calculated leveraging title and category data, we can combine these scores to create a final score.  While we have used buckets to limit the number of comparisons being performed in the steps above, there are still a lot of products we could potentially return here.  A simple function is used to limit products to a top N number of products:
# MAGIC 
# MAGIC **NOTE** You could do something similar using a row number function and a filter constraint, but we found that the function worked a bit better in this situation.

# COMMAND ----------

# DBTITLE 1,Generate Recommendations
# function to get top N product b's for a given product a
def get_top_products( data ):
  '''the incoming dataset is expected to have the following structure: id_a, id_b, score'''  
  
  rows_to_return = 11 # limit to top 10 products (+1 for self)
  
  return data.sort_values(by=['score'], ascending=False).iloc[0:rows_to_return] # might be faster ways to implement this sort/trunc

# combine similarity scores and get top N highest scoring products
recommendations = (
  category_similarity
    .join(title_similarity, on=['id_a', 'id_b'], how='inner')
    .withColumn('score', category_similarity.similarity * title_similarity.similarity)
    .select(category_similarity.id_a, category_similarity.id_b, 'score')  
    .groupBy('id_a')
      .applyInPandas(
        get_top_products, 
        schema='''
          id_a long,
          id_b long,
          score double
          ''')
    )

display(recommendations)

# COMMAND ----------

# DBTITLE 1,Clean Up Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md # Step 4b: Create Profile-Based Recommendations
# MAGIC 
# MAGIC As with our content-only recommenders, there are numerous factors that go into deciding whether recommendations should be pre-computed and cached or calculated dynamically.  Instead of sorting out the strategies one might take with regards to that, we'll focus here on defining the logic required to assemble the recommendations from the feature sets as they currently are recorded.
# MAGIC 
# MAGIC Here, we'll limit the generation of recommendations to a limited number of users and constrain our recommendations to the top 25 highest scoring items:

# COMMAND ----------

# DBTITLE 1,Define Similarity Function
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,Retrieve Datasets for Recommendations
products = (
  spark
    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
    .withColumnRenamed('features', 'features_b')
    .cache()
    )
products.count() # force caching to complete

user_profiles = (
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
    .withColumnRenamed('features', 'features_a')
    #.sample(False, 0.00005)
    ).cache()

# see how profiles distributed between buckets
display(
  user_profiles
    .groupBy('bucket')
      .agg(count('*'))
  )

# COMMAND ----------

# DBTITLE 1,Generate Recommendations
# make recommendations for sampled reviewers
recommendations = (
  products
    .hint('skew','bucket') # hint to ensure join is balanced
    .join( 
      user_profiles, 
      on='bucket', 
      how='inner'
      ) # join products to profiles on buckets
    .withColumn('score', expr('euclidean_similarity(features_a, features_b)')) # calculate similarity
    .withColumn('seq', expr('row_number() OVER(PARTITION BY reviewerID ORDER BY score DESC)'))
    .filter('seq <= 25')
    .selectExpr('reviewerID', 'id as product_id', 'seq')
    )

display(
  recommendations
  )

# COMMAND ----------

# DBTITLE 1,Clean Up Cached Datasets
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
