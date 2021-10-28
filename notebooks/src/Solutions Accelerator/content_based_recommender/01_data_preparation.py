# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the dataset we will use to explore content-based filtering recommenders.  This notebook should be run on a **Databricks ML 7.3+ cluster**.

# COMMAND ----------

# MAGIC %md # Introduction
# MAGIC 
# MAGIC Content-based recommenders enable the familiar **Related products**-type of recommendation.  These recommenders help customers identify product alternatives and better ensure that customer interactions result in sales:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC The data used to build such recommendations can also be used to build more expansive recommendations, typical of the **Based on Your Recent History**-type recommendations, which take into consideration feature preferences aligned with user feedback:
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_profile_recs2.png" width=600>

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark.sql.functions import count,  min, max, instr, monotonically_increasing_id, pandas_udf

from delta.tables import *

import pandas as pd

import gzip
import shutil

import requests
import html

# COMMAND ----------

# MAGIC %md # Step 1: Download & Decompress Files
# MAGIC 
# MAGIC The basic building block of this type of recommender is product data.  These data may include information about the manufacturer, price, materials, country of origin, *etc.* and are typically accompanied by friendly product names and descriptions.  In this series of notebooks, we'll focus on making use of the unstructured information found in product titles and descriptions as well as product category information.
# MAGIC 
# MAGIC The dataset we will use is the [2018 Amazon reviews dataset](http://deepyeti.ucsd.edu/jianmo/amazon/).  It consists of several files representing both user-generated reviews and product metadata. Focusing on the 5-core subset of data in which all users and items have at least 5 reviews, we will download the gzip-compressed files to the *gz* folder associated with a [cloud-storage mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) identified as */mnt/reviews* before decompressing the metadata JSON files to a folder named *metadata* and reviews JSON files to a folder named *reviews*.  Please note that the files associated with the *Books* category of products is being skipped as repeated requests for this file seem to be triggering download throttling from the file server:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_folder_structure2.png' width=250>
# MAGIC 
# MAGIC **NOTE** We are providing code for the downloading of these files to your storage account as similar code is supplied by the data provider.  However, you are strongly encouraged to visit the download site referenced above to review the terms and conditions for the use of this data before executing the code.  Also note that the variable *perform_download* is set to *False* to prevent the unintended downloading of data from the provider.  You'll need to change that variable's value to enable the download code in this notebook.

# COMMAND ----------

# DBTITLE 1,Download Configuration
# directories for data files
download_path = '/mnt/reviews/bronze/gz'
metadata_path = '/mnt/reviews/bronze/metadata'
reviews_path = '/mnt/reviews/bronze/reviews'

perform_download = False # set to True if you wish to redownload the gzip files

# COMMAND ----------

# DBTITLE 1,Files to Download
file_urls_to_download = [
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/AMAZON_FASHION.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_AMAZON_FASHION.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_All_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Appliances.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Appliances.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Arts_Crafts_and_Sewing.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Arts_Crafts_and_Sewing.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Automotive.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Automotive.json.gz',
  #'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz',  # skip these files to avoid overtaxing the data provider
  #'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Books.json.gz', # skip these files to avoid overtaxing the data provider
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/CDs_and_Vinyl.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_CDs_and_Vinyl.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Cell_Phones_and_Accessories.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Cell_Phones_and_Accessories.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Clothing_Shoes_and_Jewelry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Digital_Music.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Digital_Music.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Electronics.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Electronics.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Gift_Cards.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Gift_Cards.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Grocery_and_Gourmet_Food.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Grocery_and_Gourmet_Food.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Home_and_Kitchen.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Home_and_Kitchen.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Industrial_and_Scientific.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Industrial_and_Scientific.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Kindle_Store.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Kindle_Store.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Luxury_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Luxury_Beauty.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Magazine_Subscriptions.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Magazine_Subscriptions.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Movies_and_TV.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Musical_Instruments.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Musical_Instruments.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Office_Products.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Office_Products.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Patio_Lawn_and_Garden.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Patio_Lawn_and_Garden.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Pet_Supplies.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Pet_Supplies.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Prime_Pantry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Prime_Pantry.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Software.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Software.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Sports_and_Outdoors.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Sports_and_Outdoors.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Tools_and_Home_Improvement.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Tools_and_Home_Improvement.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Toys_and_Games.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Toys_and_Games.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz',
  'http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Video_Games.json.gz'
  ]

# COMMAND ----------

# DBTITLE 1,Reset Directories for Downloads
if perform_download:

  # clean up directories from prior runs
  try:
    dbutils.fs.rm(download_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(download_path)

  try:
    dbutils.fs.rm(metadata_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(metadata_path)

  try:
    dbutils.fs.rm(reviews_path, recurse=True)
  except:
    pass
  dbutils.fs.mkdirs(reviews_path)

# COMMAND ----------

# DBTITLE 1,Download & Decompress Files
if perform_download:
  
  # for each file to download:
  for file_url in file_urls_to_download:
    
    print(file_url)
    
    # extract file names from the url
    gz_file_name = file_url.split('/')[-1]
    json_file_name = gz_file_name[:-3]
    
    # determine where to place unzipped json
    if 'meta_' in json_file_name:
      json_path = metadata_path
    else:
      json_path = reviews_path

    # download the gzipped file
    request = requests.get(file_url)
    with open('/dbfs' + download_path + '/' + gz_file_name, 'wb') as f:
      f.write(request.content)

    # decompress the file
    with gzip.open('/dbfs' + download_path + '/' + gz_file_name, 'rb') as f_in:
      with open('/dbfs' + json_path + '/' + json_file_name, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

# COMMAND ----------

# MAGIC %md Let's verify we now have decompressed JSON files in our metadata and reviews folders:

# COMMAND ----------

# DBTITLE 1,List Metadata Files
display(
  dbutils.fs.ls(metadata_path)
  )

# COMMAND ----------

# DBTITLE 1,List Review Files
display(
  dbutils.fs.ls(reviews_path)
  )

# COMMAND ----------

# MAGIC %md # Step 2: Prep Metadata
# MAGIC 
# MAGIC With our metadata files in place, let's extract the relevant information from the documents and make the information more easily queriable.  In reviewing the metadata files, it appears the brand, category, title & description fields along with each product's unique identifier, it's Amazon Standard Identification Number (*asin*), will be of useful.  Quite a bit more information is available in the metadata files but we'll limit our attention to just these fields:

# COMMAND ----------

# DBTITLE 1,Prepare Environment for Data
_ = spark.sql('DROP DATABASE IF EXISTS reviews CASCADE')
_ = spark.sql('CREATE DATABASE reviews')

# COMMAND ----------

# DBTITLE 1,Extract Common Elements from Metadata JSON
# common elements of interest from json docs (only import ones actually used later)
metadata_common_schema = StructType([
	StructField('asin', StringType()),
	StructField('category', ArrayType(StringType())),
	StructField('description', ArrayType(StringType())),
	StructField('title', StringType())
	])

# read json to dataframe
raw_metadata = (
  spark
    .read
    .json(
      metadata_path,
      schema=metadata_common_schema
      )
    )

display(raw_metadata)

# COMMAND ----------

# MAGIC %md A [notebook](https://colab.research.google.com/drive/1Zv6MARGQcrBbLHyjPVVMZVnRWsRnVMpV) made available by the data host indicates some entries may be invalid and should be removed. These records are identified by the presence of the *getTime* JavaScript method call in the *title* field:

# COMMAND ----------

# DBTITLE 1,Eliminate Unnecessary Records
# remove bad records and add ID for deduplication work
metadata = (
  raw_metadata
    .filter( instr(raw_metadata.title, 'getTime')==0 ) # unformatted title
    )

metadata.count()

# COMMAND ----------

# MAGIC %md The dataset also contains a few duplicate entries based on the *asin* value:

# COMMAND ----------

# DBTITLE 1,Number of ASINs with More Than One Record
# count number of records per ASIN value
(
  metadata
  .groupBy('asin')  
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md Using an artificial id, we will eliminate the duplicates by arbitrarily selecting one record for each ASIN to remain in the dataset.  Notice we are caching the dataframe within which this id is defined in order to fix its value.  Otherwise, the value generated by *monotonically_increasing_id()* will be inconsistent during the self-join:

# COMMAND ----------

# DBTITLE 1,Deduplicate Dataset
# add id to enable de-duplication and more efficient lookups (cache to fix id values)
metadata_with_dupes = (
  metadata
    .withColumn('id', monotonically_increasing_id())
  ).cache()

# locate first entry for each asin
first_asin = (
  metadata_with_dupes
    .groupBy('asin')
      .agg(min('id').alias('id'))
  )

# join to eliminate unmatched entries
deduped_metadata = (
  metadata_with_dupes
    .join(first_asin, on='id', how='leftsemi')
  )

deduped_metadata.count()

# COMMAND ----------

# DBTITLE 1,Verify Duplicates Eliminated
# should return 0 if no duplicates
(
  deduped_metadata
  .groupBy('asin')
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md To make our next data processing steps easier to perform, we will persist the deduplicated data to storage.  By using Delta Lake as the storage format, we are enabling a set of [data modification statements](https://docs.databricks.com/delta/delta-update.html) which we will employ later:

# COMMAND ----------

# DBTITLE 1,Persist Deduplicated Data
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS reviews.metadata')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/silver/metadata', ignore_errors=True)

# persist as delta table
(
  deduped_metadata
   .repartition(sc.defaultParallelism * 4)
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/silver/metadata')
  )

# make table queriable
_ = spark.sql('''
  CREATE TABLE IF NOT EXISTS reviews.metadata
  USING DELTA
  LOCATION '/mnt/reviews/silver/metadata'
  ''')

# show data
display(
  spark.table('reviews.metadata')
  )

# COMMAND ----------

# DBTITLE 1,Drop Cached Data
_ = metadata_with_dupes.unpersist()

# COMMAND ----------

# MAGIC %md With our deduplicated data in place, let's turn now to the cleansing of some of the fields.  Taking a close look at the *description* field, we can see there are unescaped HTML characters and full HTML tags which we need to clean up.  We can see some similar cleansing is needed for the *title* and the *category* fields.  With the *category* field, it appears that the category hierarchy breaks down at a level where an HTML tag is encountered.  For that field, we will truncate the category information at the point a tag is discovered.
# MAGIC 
# MAGIC To make this code easier to implement, we'll make use of a [pandas UDF](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html) and update our data in place:

# COMMAND ----------

# DBTITLE 1,Define Function to Unescape HTML
# pandas function to unescape HTML characters
@pandas_udf(StringType())
def unescape_html(text: pd.Series) -> pd.Series:
  return text.apply(html.unescape)

# register function for use with SQL
_ = spark.udf.register('unescape_html', unescape_html)

# COMMAND ----------

# DBTITLE 1,Cleanse Titles
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT -- remove HTML from feature array
# MAGIC     a.id,
# MAGIC     unescape_html(
# MAGIC       REGEXP_REPLACE(a.title, '<.+?>', '')
# MAGIC       ) as title
# MAGIC   FROM reviews.metadata a
# MAGIC   WHERE a.title RLIKE '<.+?>|&\\w+;'  -- contains html tags & chars
# MAGIC   ) y
# MAGIC ON x.id = y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.title = y.title

# COMMAND ----------

# DBTITLE 1,Cleanse Descriptions
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT -- remove HTML from feature array
# MAGIC     a.id,
# MAGIC     COLLECT_LIST( 
# MAGIC       unescape_html(
# MAGIC         REGEXP_REPLACE(b.d, '<.+?>', '')
# MAGIC         )
# MAGIC       ) as description
# MAGIC   FROM reviews.metadata a
# MAGIC   LATERAL VIEW explode(a.description) b as d
# MAGIC   WHERE b.d RLIKE '<.+?>|&\\w+;'  -- contains html tags & chars
# MAGIC   GROUP BY a.id
# MAGIC   ) y
# MAGIC ON x.id = y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.description = y.description

# COMMAND ----------

# DBTITLE 1,Cleanse Categories
# MAGIC %sql
# MAGIC 
# MAGIC MERGE INTO reviews.metadata x
# MAGIC USING ( 
# MAGIC   SELECT  -- only keep elements prior to html
# MAGIC     m.id,
# MAGIC     COLLECT_LIST( unescape_html(o.c) ) as category
# MAGIC   FROM reviews.metadata m
# MAGIC   INNER JOIN (
# MAGIC     SELECT -- find first occurance of html in categories
# MAGIC       a.id,
# MAGIC       MIN(b.index) as first_bad_index
# MAGIC     FROM reviews.metadata a
# MAGIC     LATERAL VIEW posexplode(a.category) b as index,c
# MAGIC     WHERE b.c RLIKE '<.+?>'  -- contains html tags
# MAGIC     GROUP BY a.id
# MAGIC     ) n 
# MAGIC     ON m.id=n.id
# MAGIC   LATERAL VIEW posexplode(m.category) o as index,c
# MAGIC   WHERE o.index < n.first_bad_index
# MAGIC   GROUP BY m.id
# MAGIC   ) y
# MAGIC   ON x.id=y.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET x.category=y.category

# COMMAND ----------

# DBTITLE 1,Cleanup Delta Table
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.metadata RETAIN 0 HOURS')

# COMMAND ----------

# MAGIC %md Let's now see how our cleansed metadata appears:

# COMMAND ----------

# DBTITLE 1,Review Cleansed Metadata
display(
  spark.table('reviews.metadata')
  )

# COMMAND ----------

# MAGIC %md # Step 3: Prep Reviews
# MAGIC 
# MAGIC As with our metadata files, there are only a limited number of fields in the reviews JSON documents relevant to our needs. We'll retrieve the ASIN for each product as well as the reviewer's ID and the time of their review.  Fields such as whether a purchase was verified or the number of other users who found the review could be useful but we'll leave them be for now:

# COMMAND ----------

# DBTITLE 1,Extract Common Elements from Reviews JSON
# common elements of interest from json docs
reviews_common_schema = StructType([
  StructField('asin', StringType()),
  StructField('overall', DoubleType()),
  StructField('reviewerID', StringType()),
  StructField('unixReviewTime', LongType())
  ])

# read json to dataframe
reviews = (
  spark
    .read
    .json(
      reviews_path,
      schema=reviews_common_schema
      )
    )

# present data for review
display(
  reviews
  )

# COMMAND ----------

# MAGIC %md A quick check for duplicates finds that we're have a little clean up to do.  While maybe not truly duplicates, if a user submits multiple reviews on a single product, we want to take the latest of these as their go-forward review. From there, we want to make sure the system is not capturing multiple records for that same, last date and time:

# COMMAND ----------

# DBTITLE 1,Deduplicate Reviews
# tack on sequential ID
reviews_with_duplicates = (
  reviews.withColumn('rid', monotonically_increasing_id())
  ).cache() # cache to fix the id in place


# locate last product review by reviewer
last_review_date = (
  reviews_with_duplicates
    .groupBy('asin', 'reviewerID')
      .agg(max('unixReviewTime').alias('unixReviewTime'))
  )

# deal with possible multiple entries on a given date
last_review = (
  reviews_with_duplicates
    .join(last_review_date, on=['asin','reviewerID','unixReviewTime'], how='leftsemi')
    .groupBy('asin','reviewerID')
      .agg(min('rid').alias('rid'))
    )

# locate last product review by a user
deduped_reviews = (reviews_with_duplicates
  .join(last_review, on=['asin','reviewerID','rid'], how='leftsemi')
  .drop('rid')
  )

display(deduped_reviews)

# COMMAND ----------

# MAGIC %md Let's now persist our data to a Delta Lake table before proceeding:

# COMMAND ----------

# DBTITLE 1,Persist Deduplicated Data
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS reviews.reviews')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/silver/reviews', ignore_errors=True)

# persist reviews as delta table
(
  deduped_reviews
   .repartition(sc.defaultParallelism * 4)
   .write 
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/silver/reviews')
  )

# make table queriable
_ = spark.sql('''
  CREATE TABLE IF NOT EXISTS reviews.reviews
  USING DELTA
  LOCATION '/mnt/reviews/silver/reviews'
  ''')

# COMMAND ----------

# DBTITLE 1,Drop Cached Dataset
_ = reviews_with_duplicates.unpersist()

# COMMAND ----------

# MAGIC %md To make joining to the product metadata easier, we'll add the product ID generated earlier with our metadata to our reviews table:

# COMMAND ----------

# DBTITLE 1,Update Reviews with Product IDs
_ = spark.sql('ALTER TABLE reviews.reviews ADD COLUMNS (product_id long)')

# retrieve asin-to-id map
ids = spark.table('reviews.metadata').select('asin','id')

# access reviews table in prep for merge
reviews = DeltaTable.forName(spark, 'reviews.reviews')

# perform merge to update ID
( reviews.alias('reviews')
    .merge(
      ids.alias('metadata'),
      condition='reviews.asin=metadata.asin'
      )
    .whenMatchedUpdate(set={'product_id':'metadata.id'})
).execute()

# display updated records
display(
  spark.table('reviews.reviews')
  )

# COMMAND ----------

# DBTITLE 1,Cleanup Delta Table
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.reviews RETAIN 0 HOURS')
