# Databricks notebook source
# MAGIC %md
# MAGIC # Chapter5. 協調フィルタリングのデプロイ

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to explore how the collaborative-filter recommenders developed in previous notebooks might be operationalized. This notebook should be run on a **Databricks ML 7.1+ cluster**.

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC どのレコメンデーションシステムも、そのベースとなるデータのボリュームやボラティリティに応じて、導入パターンが決まってきます。また、レコメンデーションで実現する特定のビジネスシナリオは、レコメンデーション・プラットフォームの実装方法にも影響します。このノートブックでは、一般的なシナリオに沿った方法で、ユーザーベースおよびアイテムベースの協調フィルタを展開する仕組みを探っていきます。ただし、決してここで示した通りにユーザーベースまたはアイテムベースのレコメンダーを展開すべきだと提案しているわけではありません。 レコメンデーションシステムを導入する際には、レコメンデーションを使用するアプリケーションの開発者やアーキテクト、そして同時に、レコメンデーションがもたらす成果に責任を持つビジネスステークホルダーと話し合うことを強くお勧めします。 このことを念頭に置いて、このノートブックで検討した展開パターンを簡単に見てみましょう。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_pipeline.png" width="700">
# MAGIC 
# MAGIC 今回の展開シナリオでは、毎日、毎週、あるいは毎月、ユーザーや商品のペアを生成するプロセスを想定しています。 これらのペアは、評価データや商品データとともに、リレーショナルデータベース（またはその他のデータストア）に複製され、調整されたクエリを使用してアプリケーションで提示されるレコメンデーションを生成します。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのimport
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

# MAGIC %md # Step 1a: ユーザーペアを構成
# MAGIC 
# MAGIC ユーザーベースの協調フィルタ推薦を生成するための最初のステップは、各ユーザーの評価ベクトルを収集し、各ベクトルをLSHバケットに割り当てることです。 多くのソリューションでは、このようなステップは[パイプライン](https://spark.apache.org/docs/latest/ml-pipeline.html)として構成されますが、ここで想定しているバッチシナリオでは、スクリプトやノートブック内でコードを維持する方が簡単かもしれません。(LSHにパイプラインを採用する例として、最終的なコンテンツベースレコメンダーのノートブックを参照してください)
# MAGIC 
# MAGIC まず始めに、ソリューションの対象となるユーザーの詳細を取得する必要があります。ここでは、校正期間中のユーザーデータをすべて活用しています。 実際には、購買パターンの季節的変動を考慮し、推奨を行う将来の期間に関連する1つ以上の過去の期間のデータを使用して、このデータセットを制限することができます。

# COMMAND ----------

# DBTITLE 1,ユーザーの評価を取得してレコメンデーションを構築する
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

# DBTITLE 1,レーティングを特徴量ベクターに変換
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

# DBTITLE 1,ユーザーベクターへLSHバケットを割り当てる
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

# MAGIC %md 
# MAGIC 
# MAGIC 前のセルの最後のステップでは、LSHでバケットされたデータをディスクに保存します。 これにより、長時間実行される次のステップの実行中に問題が発生した場合に、再起動するための一貫した基準を得ることができます。
# MAGIC 
# MAGIC 次のステップでは、`approxSimiliarityJoin()`を使用して、ユーザー集団のサブセットから特定の距離にいるユーザーを返します。閾値をかなり高く設定して多数のユーザを返すようにしますが、直近のクエリを実行して得られる結果を模して、最も類似した10人のユーザ（およびターゲット・ユーザ自身）に限定します。 データ検索の問題を、以前のノートブックで検討した`approxNearnestNeighbor()`メソッドではなく、この方法でアプローチしている理由は、類似性結合が複数のユーザーのデータを検索するために構築されているためであり（最近のネイバーズテクニックによる単一ユーザーの検索とは対照的です）、このアプローチは非常に高速です。

# COMMAND ----------

# DBTITLE 1,ユーザーペアをトップ10に限定する関数を定義
def get_top_users( data ):
  '''the incoming dataset is expected to have the following structure: user_a, user_b, distance''' 
  
  rows_to_return = 11 # limit to top 10 users (+1 for self)
  min_score = 1 / (1+math.sqrt(2))
  
  data['sim']= 1/(1+data['distance'])
  data['similarity']= (data['sim']-min_score)/(1-min_score)
  ret = data[['user_id', 'paired_user_id', 'similarity']]
  
  return ret.sort_values(by=['similarity'], ascending=False).iloc[0:rows_to_return] # might be faster ways to implement this sort/trunc

# COMMAND ----------

# DBTITLE 1,出力先のテーブルを初期化しておく
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

# DBTITLE 1,ユーザーペアの生成
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

# DBTITLE 1,ユーザーペアを確認
display(
  spark.table('instacart.user_pairs')
  )

# COMMAND ----------

# MAGIC %md # Step 1b: 製品ペアを構成する
# MAGIC 
# MAGIC アイテムベースの協調フィルタを構築するための最初のステップは、過去の関連期間の製品ペアを組み立てることです。 ユーザーペアの生成とは異なり、バケット化技術は使用せず、代わりにデータで実際に観察されたペアのブルートフォース生成を行います。 実行される計算の数を制限するために、製品Aと製品Bの非冗長な比較を実行し、自己比較、*すなわち*製品Aと製品Aの比較データとともに、反転した製品Bと製品Aの比較を出力テーブルに単純に挿入します。このパターンは前回のノートでも検討しました。
# MAGIC 
# MAGIC 前回のノートとは異なり、6人未満のユーザーに関連する製品ペアを排除するためのフィルターを実装していることに注目してください。 前作では、データセットの理想的な設定を検討するために、レコメンデーションの生成時にフィルターを実装しました。 今回は、データ処理とクエリの時間を短縮するために、フィルターをデータ処理パイプラインのさらに上流に移動させています。

# COMMAND ----------

# DBTITLE 1,製品ペアの類似性を算出
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

# DBTITLE 1,出力先のテーブルを初期化しておく
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

# DBTITLE 1,プロダクトペアをDeltaに書き出す(永続化)
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

# MAGIC %md # Step 2a: ユーザーベースのレコメンデーションを構築する
# MAGIC 
# MAGIC ユーザーペアが揃ったところで、次は実際のレコメンデーションの生成に入ります。 ほとんどのレコメンデーションのシナリオでは、検索の速さが重要です。 同時に、データセット内のユーザーのうち何人が、データがアクティブである間にレコメンデーションエンジンを利用する可能性があるかを考慮する必要があります。レコメンデーションを利用するユーザーの数がデータセット内のユーザー総数に比べて少ない場合、レコメンデーションを動的に生成する（そしておそらく後で再利用するためにキャッシュする）ことを検討することができます。
# MAGIC 
# MAGIC レコメンデーションを動的に生成するためには、リレーショナル・データベース・エンジンの採用が考えられます。 高いクエリー・パフォーマンスのSLAを満たすために、これらの技術で利用可能なパーティショニングやインデックス戦略を採用することができます。 また、データの一部を非正規化することで、一貫した検索速度を確保することができます。
# MAGIC 
# MAGIC 
# MAGIC このコードを実行するための依存関係が増えるため、このノートブックでは特定のRDMBSにデータをパブリッシュする方法を説明しません。 一般的なRDMBSへのデータ公開に関する情報やコードサンプルは、以下のリンクから入手できます。</p>
# MAGIC 
# MAGIC * [Azure SQL Server](https://docs.microsoft.com/en-us/sql/connect/spark/connector?view=sql-server-ver15) 
# MAGIC * [AWS RDS & other JDBC or ODBC-supported RDMBSs](https://docs.databricks.com/data/data-sources/sql-databases.html)
# MAGIC 
# MAGIC このノートでは、代わりにDelta Lake対応のテーブルに対してクエリを実行します。 採用する主なテーブルは以下の通りです。</p>
# MAGIC 1. User-Pairs
# MAGIC 2. User-Ratings
# MAGIC 3. Products
# MAGIC 
# MAGIC これらのテーブルのデータを以下のようにして持ってきます。

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

# MAGIC %md
# MAGIC 
# MAGIC クエリのパフォーマンスは理想的ではありませんが、ここではパフォーマンスではなくロジックに重点を置いています。 必要なパフォーマンスを得るためには、このクエリのベースとなるテーブルをRDBMSに複製し、インデックスを作成してアクセスを高速化するのが最適です。

# COMMAND ----------

# DBTITLE 1,Release Cached Objects
# MAGIC %sql 
# MAGIC UNCACHE TABLE cached__user_ratings;
# MAGIC UNCACHE TABLE cached__products

# COMMAND ----------

# MAGIC %md # Step 2b: アイテムベースのレコメンデーションを構築する
# MAGIC 
# MAGIC ユーザーベースのレコメンデーションと同様に、レコメンデーションのダイナミックな生成を可能にするために、通常、いくつかの基本テーブルをRDMBSに公開します。 このために必要なテーブルは</p>
# MAGIC 
# MAGIC 1. Product-Pairs
# MAGIC 2. User-Ratings
# MAGIC 3. Products
# MAGIC 
# MAGIC これらのテーブルを使用して、推薦生成ロジックをカプセル化するための汎用ビューを以下のように構築することができます。

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

# MAGIC %md
# MAGIC 
# MAGIC ここでも、これらのオブジェクトをRDBMSに複製することで、パフォーマンスを実現しています。 ここでのクエリは、これらのデータが動的に組み合わされて推奨事項を形成する手段を示しているに過ぎません。
