# Databricks notebook source
# MAGIC %md このノートブックの目的は、コンテンツベースのフィルタリングレコメンダーを調査するために使用するデータセットを準備することです。 このノートブックは **Databricks ML 7.3+ クラスタ** 上で実行する必要があります。

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC コンテンツベースのレコメンダーは、おなじみの**関連商品**タイプのレコメンデーションを可能にします。 これらのレコメンダーは、お客様が製品の代替品を特定するのに役立ち、お客様との対話がより確実に売上に結びつくようにします。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC このようなレコメンデーションを構築するために使用されたデータは、より広範囲なレコメンデーションを構築するために使用することもできます。典型的な**Based on Your Recent History**タイプのレコメンデーションでは、ユーザーのフィードバックに沿った機能の好みを考慮します。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_profile_recs2.png" width=600>

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
from pyspark.sql.types import *
from pyspark.sql.functions import count,  min, max, instr, monotonically_increasing_id, pandas_udf

from delta.tables import *

import pandas as pd

import gzip
import shutil

import requests
import html

# COMMAND ----------

# MAGIC %md # Step 1: ファイルのダウンロードと解凍
# MAGIC 
# MAGIC この種のレコメンダーの基本的な構成要素は、製品データです。 これらのデータには、メーカー、価格、素材、原産国などの情報が含まれており、一般的には親しみやすい商品名や説明文が添えられています。 この連載では、商品のタイトルや説明文に含まれる非構造化情報や、商品のカテゴリー情報を活用することに焦点を当てていきます。
# MAGIC 
# MAGIC 今回使用するデータセットは、[2018 Amazon reviews dataset](http://deepyeti.ucsd.edu/jianmo/amazon/)です。 このデータセットは、ユーザーが作成したレビューと商品のメタデータの両方を表す複数のファイルから構成されています。すべてのユーザーとアイテムが少なくとも5つのレビューを持つ5コアのサブセットに焦点を当て、gzip圧縮されたファイルを、*/mnt/reviews*と名付けられた[cloud-storage mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)に関連付けられた*gz*フォルダにダウンロードしてから、メタデータのJSONファイルを*metadata*というフォルダに、レビューのJSONファイルを*reviews*というフォルダに解凍していきます。 なお、製品の*Books*カテゴリーに関連するファイルはスキップされています。これは、このファイルへの繰り返しのリクエストが、ファイルサーバーからのダウンロードを制限する原因になっているようです。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_folder_structure2.png' width=250>
# MAGIC 
# MAGIC **注意** データ提供者から同様のコードが提供されているため、お客様のストレージ・アカウントにこれらのファイルをダウンロードするためのコードを提供しています。 ただし、コードを実行する前に、上記のダウンロードサイトにアクセスして、このデータの使用に関する条件を確認することを強くお勧めします。 また、データ提供者から意図せずにデータがダウンロードされることを防ぐために、変数*perform_download*が*False*に設定されていることに注意してください。 このノートブックのダウンロードコードを有効にするには、この変数の値を変更する必要があります。

# COMMAND ----------

# DBTITLE 1,Configurationのダウンロード
# directories for data files
download_path = '/mnt/reviews/bronze/gz'
metadata_path = '/mnt/reviews/bronze/metadata'
reviews_path = '/mnt/reviews/bronze/reviews'

perform_download = False # set to True if you wish to redownload the gzip files

# COMMAND ----------

# DBTITLE 1,ファイルのDownload
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

# DBTITLE 1,ダウンロードディレクトリの初期化
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

# DBTITLE 1,ダウンロード・解答
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

# MAGIC %md 解凍されたJSONファイルがmetadataとreviewsフォルダにあることを確認しましょう。

# COMMAND ----------

# DBTITLE 1,メタデータの確認(List)
display(
  dbutils.fs.ls(metadata_path)
  )

# COMMAND ----------

# DBTITLE 1,List Review Files
display(
  dbutils.fs.ls(reviews_path)
  )

# COMMAND ----------

# MAGIC %md # Step 2: Metadataの準備
# MAGIC 
# MAGIC メタデータファイルを用意したら、ドキュメントから関連情報を抽出し、情報をより簡単に検索できるようにしましょう。 メタデータファイルを見てみると、ブランド、カテゴリー、タイトル、説明の各フィールドと、各商品のユニークな識別子であるAmazon Standard Identification Number (*asin*)が役に立ちそうです。 メタデータファイルには他にも多くの情報が含まれていますが、ここではこれらのフィールドだけに注目します。

# COMMAND ----------

# DBTITLE 1,データのための環境設定
_ = spark.sql('DROP DATABASE IF EXISTS reviews CASCADE')
_ = spark.sql('CREATE DATABASE reviews')

# COMMAND ----------

# DBTITLE 1,メタデータJSONから共通要素を抽出
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

# MAGIC %md データホストが提供する[ノートブック](https://colab.research.google.com/drive/1Zv6MARGQcrBbLHyjPVVMZVnRWsRnVMpV)は、いくつかのエントリが無効である可能性があり、削除する必要があることを示しています。これらのレコードは、*title*フィールドに*getTime*というJavaScriptのメソッドコールがあることで識別されます。

# COMMAND ----------

# DBTITLE 1,不要なレコードを削除
# remove bad records and add ID for deduplication work
metadata = (
  raw_metadata
    .filter( instr(raw_metadata.title, 'getTime')==0 ) # unformatted title
    )

metadata.count()

# COMMAND ----------

# MAGIC %md また、このデータセットには、「*asin*」の値に基づいて、いくつかの重複したエントリが含まれています。

# COMMAND ----------

# DBTITLE 1,2つ以上のレコードを持つASINの数
# count number of records per ASIN value
(
  metadata
  .groupBy('asin')  
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md 人工的なIDを使って、各ASINのレコードを任意に1つ選び、データセットに残すことで、重複を排除します。 このidが定義されたデータフレームをキャッシュして、その値を固定していることに注意してください。 そうしないと、*monotonically_increasing_id()*で生成された値は、self-joinの際に矛盾してしまいます。

# COMMAND ----------

# DBTITLE 1,データセットの重複排除
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

# DBTITLE 1,重複排除の確認
# should return 0 if no duplicates
(
  deduped_metadata
  .groupBy('asin')
    .agg(count('*').alias('recs'))
  .filter('recs > 1')
  ).count()

# COMMAND ----------

# MAGIC %md 次のデータ処理を簡単に行うために、重複排除されたデータをストレージに永続化する。 ストレージのフォーマットとしてDelta Lakeを使用することで、後に使用する[data modification statement](https://docs.databricks.com/delta/delta-update.html)のセットが可能になります。

# COMMAND ----------

# DBTITLE 1,重複排除の永続化
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

# DBTITLE 1,キャッシュの削除
_ = metadata_with_dupes.unpersist()

# COMMAND ----------

# MAGIC %md 重複排除されたデータができたので、次にいくつかのフィールドのクレンジングをしてみましょう。 description*フィールドをよく見てみると、エスケープされていないHTML文字や完全なHTMLタグがあり、これをクリーンアップする必要があります。 また、*title*と*category*フィールドにも同様のクレンジングが必要です。 category*フィールドでは、HTMLタグが発生するレベルでカテゴリ階層が壊れているように見えます。 このフィールドでは、タグが検出された時点でカテゴリ情報を切り詰めます。
# MAGIC 
# MAGIC このコードを簡単に実装するために、[pandas UDF](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html)を利用し、その場でデータを更新することにします。

# COMMAND ----------

# DBTITLE 1,Define Function to Unescape HTML
# pandas function to unescape HTML characters
@pandas_udf(StringType())
def unescape_html(text: pd.Series) -> pd.Series:
  return text.apply(html.unescape)

# register function for use with SQL
_ = spark.udf.register('unescape_html', unescape_html)

# COMMAND ----------

# DBTITLE 1,タイトルのクレンジング
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

# DBTITLE 1,Descriptionsのクレンジング
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

# DBTITLE 1,Categoriesのクレンジング
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

# DBTITLE 1,Delta Tableのクリーンアップ
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.metadata RETAIN 0 HOURS')

# COMMAND ----------

# MAGIC %md それでは、クレンジングされたメタデータがどのように表示されるか見てみましょう。

# COMMAND ----------

# DBTITLE 1,クレンジング済みMetadataの確認
display(
  spark.table('reviews.metadata')
  )

# COMMAND ----------

# MAGIC %md # Step 3: データ準備のレビュー
# MAGIC 
# MAGIC メタデータファイルと同様に、レビューのJSONドキュメントには、私たちのニーズに関連する限られた数のフィールドしかありません。私たちは、各製品のASIN、レビュアーのID、レビューの時間を取得します。 購入が確認されたかどうかや、レビューを見つけた他のユーザーの数などのフィールドは有用かもしれませんが、今のところはそのままにしておきます。

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

# MAGIC %md 重複していないかどうかを確認したところ、少し整理しなければならないことがわかりました。 本当の意味での重複ではないかもしれませんが、ユーザーが1つの製品に複数のレビューを投稿した場合、最新のレビューを今後のレビューとして採用したいと考えています。そこから、システムが同じ最終日時の複数の記録を取得していないことを確認したいのです。

# COMMAND ----------

# DBTITLE 1,重複排除の確認
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

# MAGIC %md 先に進む前に、データをDelta Lakeのテーブルに永続化させましょう。

# COMMAND ----------

# DBTITLE 1,重複排除の永続化
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

# DBTITLE 1,キャッシュの削除
_ = reviews_with_duplicates.unpersist()

# COMMAND ----------

# MAGIC %md 商品のメタデータとの結合を容易にするために、先にメタデータで生成した商品IDをレビューテーブルに追加します。

# COMMAND ----------

# DBTITLE 1,Product IDsを使ってReviewsをアップデート
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

# DBTITLE 1,Delta Tableのクリーンアップ
spark.conf.set('spark.databricks.delta.retentionDurationCheck.enabled', False)
_ = spark.sql('VACUUM reviews.reviews RETAIN 0 HOURS')
