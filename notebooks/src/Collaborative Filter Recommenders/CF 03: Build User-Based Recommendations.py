# Databricks notebook source
# MAGIC %md The purpose of this notebook is to build and evaluate user-based collaborative filtering recommendations.  This notebook is designed to run on a **Databricks 7.1+ cluster**.

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC ユーザーマッチングを行うための基礎ができたところで、ユーザー間の製品購入の類似性を利用した協調フィルタを構築してみましょう。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_userbasedcollab.gif" width="300">

# COMMAND ----------

# DBTITLE 1,必要なライブラリのimport
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, max, collect_list, lit, expr, coalesce, pow, sum
from pyspark.sql.types import *

import pandas as pd
import numpy as np

import math
import shutil

# COMMAND ----------

# MAGIC %md # Step 1: レコメンデーションの構築
# MAGIC 
# MAGIC レコメンデーションを構築するには、前回のノートブックで検討した評価ベクトルとLSHデータセットを再構築する必要があります。そこから始めていきましょう。

# COMMAND ----------

# DBTITLE 1,評価値ベクターを構成する
# define and register UDF for vector construction
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    ind, val = zip(*sorted(zip(index_list, value_list)))
    return Vectors.sparse(size, ind, val)
    
_ = spark.udf.register('to_vector', to_vector)

# generate ratings vectors 
ratings_vectors = spark.sql('''
  SELECT 
      user_id,
      to_vector(size, index_list, value_list) as ratings,
      size,
      index_list,
      value_list
    FROM ( 
      SELECT
        user_id,
        (SELECT max(product_id) + 1 FROM instacart.products) as size,
        COLLECT_LIST(product_id) as index_list,
        COLLECT_LIST(normalized_purchases) as value_list
      FROM ( -- all users, ratings
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE split = 'calibration'
        )
      GROUP BY user_id
      )
    ''')

# COMMAND ----------

# MAGIC %md 
# MAGIC **注意** 
# MAGIC ここで使用している*bucketLength*と*numHashTable*の設定は、前回のノートブックで検討したものとは異なります。 これらの設定は、パフォーマンスと評価指標を考慮した結果です。(以下の通りです)

# COMMAND ----------

# DBTITLE 1,LSHデータセットを構築する
bucket_length = 0.0025
lsh_tables = 5

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

hashed_vectors.createOrReplaceTempView('hashed_vectors')

# COMMAND ----------

# MAGIC %md 
# MAGIC ここでは、ある顧客に対してどのようにレコメンデーションを作成するかを考えてみましょう。 例えば、顧客の一人である`user_id=148`の値を使って、製品の好みのベクトルを組み立てることができます。

# COMMAND ----------

# DBTITLE 1,一つのユーザーベクターを見てみる
user_148 = (
  hashed_vectors
    .filter('user_id=148')
    .select('user_id','ratings')
  )

user_148.collect()[0]['ratings']

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC このお客様に対しては、推薦の対象となりうる類似のお客様を特定する必要があります。 そのためには、あるユーザーと最も類似しているユーザーの目標数を指定する方法があります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_nearestneighbors.gif" width="300">
# MAGIC 
# MAGIC この *一番近い隣人* のアプローチは、評価を構築するための一貫した数のユーザーを返すという利点がありますが、それらのユーザーと我々の与えられたユーザーとの間の類似性は変化する可能性があります。
# MAGIC 
# MAGIC **注意** ユーザー自体、ここではuser_id 148、は結果セットに含まれます。 もし、10人の*他の*ユーザーを検索したい場合は、11人の最近傍のユーザーを指定し、その後、結果から自分のユーザーを除外する必要があります。

# COMMAND ----------

# DBTITLE 1,類似ユーザーTop10を抽出する
number_of_customers = 10

# retrieve n nearest customers 
similar_k_users = (
  fitted_lsh.approxNearestNeighbors(
    hashed_vectors, 
    user_148.collect()[0]['ratings'], # must be a vector value (not a dataframe)
    number_of_customers, 
    distCol='distance'
    )
    .select('user_id', 'distance')
  )
  
display(similar_k_users)

# COMMAND ----------

# MAGIC %md 
# MAGIC 類似したユーザーを特定する問題に取り組むもう一つの方法は、与えられたユーザーからの最大距離を定義し、その範囲内のすべてのユーザーを選択することです。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_distance.gif" width="400">
# MAGIC 
# MAGIC このアプローチでは、より予測可能な類似度を持つユーザーを受け取ることができますが、返されるユーザーの数が（もしあれば）変動することに対処しなければなりません。 また、データセットを調査して、どの距離の*しきい値*が望ましい数のユーザーを一貫して提供するかを理解する必要があります。
# MAGIC 
# MAGIC **注意** `approxSimilarityJoin()`メソッドでは、対象となるユーザーをデータフレームとして送信する必要があります。この利点は、ターゲットユーザーのデータセットに複数のユーザーを提供できることです（ここでは示していませんが、後のノートブックで採用されます）。

# COMMAND ----------

# DBTITLE 1,指定した距離範囲のユーザーを抽出する
max_distance_from_target = 1.3

# retreive all users within a distance range
similar_d_users = (
    fitted_lsh.approxSimilarityJoin(
      user_148,
      hashed_vectors,  
      threshold = max_distance_from_target, 
      distCol='distance'
      )
    .selectExpr('datasetA.user_id as user_a', 'datasetB.user_id as user_b', 'distance')
    .orderBy('distance', ascending=True)
    )
  
display(similar_d_users)

# COMMAND ----------

# MAGIC %md 
# MAGIC どのようにして「似ている」ユーザーを選ぶかは別にして、類似性がどのように計算されるかを理解することが重要です。 このようなベクトル空間では、類似性は多くの場合、2つのベクトル間の距離に基づいています。距離を計算するにはいくつかの方法がありますが、ユークリッド距離が最も一般的なものの1つです。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_distance.png" width="400">
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_basicsimilarity.png" width="350">
# MAGIC 
# MAGIC 2人のユーザーの距離が遠いほど、似ていないことになります。 この点から、類似性は距離そのものの逆数と考えることができます。 しかし、距離はゼロになることもあるので、ゼロ除算のエラーを避けるために、距離に1を加えて関係を修正することが多い。 
# MAGIC 
# MAGIC 類似性を計算する方法は他にもたくさんありますが、私たちの目的にはこの方法が適しているようです。

# COMMAND ----------

# DBTITLE 1,類似ユーザーとの類似度を算出する
# calculate similarity score
similar_users = (
  similar_k_users
    .withColumn('similarity', lit(1) / (lit(1) + col('distance')))
  )

display(similar_users)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC L2正規化ベクトル空間では、2点間の最小距離は0.0で、最大距離は2の平方根に相当します。 これは、潜在的な類似性スコアの最大値が1.0、最小値が0.414であることを意味します。 標準的な[min-max変換](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization))を適用すると、類似度スコアを1.0(最も似ている)から0.0(最も似ていない)の範囲に変換することができます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_similarityminmax2.png" width="600">

# COMMAND ----------

# DBTITLE 1,Apply Min-Max Transformation to Similarity Score
# calculate lowest possible unscaled similarity score
min_score = 1 / (1 + math.sqrt(2))

# calculate similarity score
similar_users = (
  similar_users
    .withColumn(
       'similarity_rescaled', 
       (col('similarity') - lit(min_score)) / lit(1.0 - min_score)
       )
     )

# make available for SQL query
similar_users.createOrReplaceTempView('similar_users')

display(similar_users)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Rescaling our similarity scores makes it easier for us to judge the degree of similarity between users. To calculate a recommendation, the similarity score will be used as a weight in a weighted average of product preferences/ratings.  There's no reason we can't make further adjustments to our similarity calculations, such as squaring the rescaled similarity scores so that weight increases exponentially as similarity approaches 1.0 Such adjustments allow us to adjust the degree of influence that similarity has on our recommendations.
# MAGIC 
# MAGIC 類似性スコアをリスケーリングすることで、ユーザー間の類似性の度合いを判断しやすくなります。お勧め商品を計算する際、類似性スコアは、商品の好みや評価の加重平均の重みとして使用されます。 また、類似度の計算にさらなる調整を加えられない理由はありません。例えば、再スケーリングされた類似度スコアを2乗して、類似度が1.0に近づくにつれて重みが指数関数的に増加するようにします。このような調整により、類似度が推薦に与える影響の度合いを調整することができます。
# MAGIC 
# MAGIC 推薦スコアを計算する際には、あるユーザーが製品に対して評価/推奨をしていない場合、評価/推奨が0.0であることを意味するように使われる可能性があることを認識することが重要です。 目的に応じて、このようなロジックを適用するか、あるいはスキップするかを選択します（ユーザーに推奨される製品の範囲を広げる効果があります）。 ここでは、どの製品が（どの類似ユーザーによって）どのような暗黙の評価や類似度のスコアで推奨されているかを見ることができます。

# COMMAND ----------

# DBTITLE 1,Product Ratings from Similar Users
similar_ratings = spark.sql('''
      SELECT
        m.user_id,
        m.product_id,
        COALESCE(n.normalized_purchases, 0.0) as normalized_purchases,
        m.similarity_rescaled
      FROM ( -- get complete list of products across similar users
        SELECT
          x.user_id,
          y.product_id,
          x.similarity_rescaled
        FROM (
          SELECT user_id, similarity_rescaled
          FROM similar_users
          ) x
        CROSS JOIN instacart.products y
        ) m
      LEFT OUTER JOIN ( -- retrieve ratings actually provided by similar users
        SELECT x.user_id, x.product_id, x.normalized_purchases 
        FROM instacart.user_ratings x 
        LEFT SEMI JOIN similar_users y 
          ON x.user_id=y.user_id 
        WHERE x.split = 'calibration'
          ) n
        ON m.user_id=n.user_id AND m.product_id=n.product_id
      ''')

display(similar_ratings)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC これらの評価と類似性スコアを用いて、類似したユーザーからの類似性加重評価を平均し、製品ごとの加重平均を算出することができます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_ratingcalc.png" width="600">

# COMMAND ----------

# DBTITLE 1,Calculate Recommendation Scores
product_ratings = ( 
   similar_ratings
    .groupBy('product_id')
      .agg( 
        sum(col('normalized_purchases') * col('similarity_rescaled')).alias('weighted_rating'),
        sum('similarity_rescaled').alias('total_weight')
        )
    .withColumn('recommendation_score', col('weighted_rating')/col('total_weight'))
    #.select('product_id', 'recommendation_score')
    .orderBy('recommendation_score', ascending=False)
  )

display(product_ratings)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 類似ユーザーが商品の推奨に与える影響を見るために、対象ユーザーである*user_id 148*の評価を取得してみましょう。 このユーザーの暗黙の嗜好と推薦スコアを比較すると、類似ユーザーが推薦に与える影響がわかります。

# COMMAND ----------

# DBTITLE 1,Compare Recommendation Scores to User-Implied Scores
# retreive actual ratings from this user
user_product_ratings = (
  spark
    .table('instacart.user_ratings')
    .filter("user_id = 148 and split = 'calibration'")
  )

# combine with recommender ratings
product_ratings_for_user = (
    product_ratings
      .join( user_product_ratings, on='product_id', how='outer')
      .selectExpr('product_id', 'COALESCE(normalized_purchases, 0.0) as user_score', 'recommendation_score')
      .orderBy('recommendation_score', ascending=False)
  )

display(product_ratings_for_user)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ユーザーの実際のスコアとレコメンデーションのスコアを比較すると、レコメンデーションがユーザーのスコアと一致している場合や、似たようなユーザーの意見に基づいて他の製品が高くなったり低くなったりしている場合、また、レコメンデーションによって新製品が紹介されている場合（ユーザーのスコアが0.0で、レコメンデーションのスコアが0.0より大きい場合）など、興味深い点があります。どのユーザーがこのレコメンドに貢献するか、類似性をどのように適用するかを決めるには、数多くの小さな選択肢がありました。 また、ターゲットユーザーが購入した商品については、類似ユーザーの評価を考慮せずにそのままスコアを付与したり、過去に購入した商品を単純に除外して新しい商品だけを推奨するなど、選択肢はまだまだあります。 これらの選択肢の中で重要なのは、レコメンダーを通してどのようなビジネス成果をもたらしたいかということです。

# COMMAND ----------

# MAGIC %md 
# MAGIC # Step 2: レコメンデーションの評価
# MAGIC 
# MAGIC ここまでで、ユーザーベースの協調フィルタによる推薦を計算する仕組みはできましたが、作成された推薦は良いものでしょうか？これは、レコメンダーを使用する際に考慮すべき重要なポイントであり、「*良い*とは何か？」
# MAGIC 
# MAGIC 
# MAGIC [Gunawardana and Shani](https://www.jmlr.org/papers/volume10/gunawardana09a/gunawardana09a.pdf)は、この問題について優れた考察を行っています。彼らは、レコメンダーは特定の目標を達成するために存在し、その目標を達成する能力の観点から評価されるべきだと主張しています。 では、**私たちのレコメンダーの*目標*は何ですか**？
# MAGIC 
# MAGIC 今回のデータセットのような食料品に関するシナリオでは、ユーザーが購入したいと思うような食料品の選択肢を提示することが最も重要な目的です。
# MAGIC 
# MAGIC 1. Enable a more efficient shopping experience,
# MAGIC 2. Encourage the customer to buy new products, *i.e.* products they have not purchased from us in the past,
# MAGIC 3. Encourage the customer to buy old products, *i.e.* products they have purchased from us in the past, but which they had not originally intended to purchase when coming to the application, site or store
# MAGIC 
# MAGIC これらの結果に対するレコメンダーの影響は、実験的な設定で簡単に測定できます。あるユーザーにはレコメンダーが表示され、あるユーザーには表示されません。 例えば、私たちの目標がより速い買い物体験を可能にすることだとしたら、私たちのレコメンダーに接したお客様は、接していないお客様よりも速く入口からチェックアウトまでの旅を終えるのでしょうか？ お客様に新しい商品を購入していただくことが目的の場合、レコメンダーに接触したお客様は、接触していないお客様に比べて、過去の購入履歴にないおすすめ商品を高い確率で購入するのでしょうか？ 我々の目標が、最初のエンゲージメントのきっかけとはならなかったかもしれない、以前に購入した商品の購入を促すことである場合、我々のレコメンダーに接触した顧客は、接触していない顧客に比べて、（以前に購入した商品に限定した）全体のバスケットサイズが大きくなるか？
# MAGIC 
# MAGIC 概念的には簡単に測定できますが、このような（オンライン）実験を実施するには、慎重な計画と実施が必要であり、機会費用がかかります。また、お客様のブランドに対する認識に悪影響を与えるような悪い顧客体験を提供するリスクもあります。このような理由から、私たちはしばしば、レコメンダーの現実世界の目標を、実験的な展開を試みる前にオフラインで評価できる代理目標に置き換えています。
# MAGIC 
# MAGIC オフライン評価のための代理目標を定義する際には、レコメンドスコアが何を表しているのかをよく考えてみるとよいでしょう。 これらのワークブックで紹介されているシナリオでは、私たちのスコアは、リピート購入から得られる暗黙の評価の加重平均値です。 Hu, Koren & Volinsky](http://yifanhu.net/PUB/cf.pdf)は、消費から得られる暗黙の評価について非常に優れた研究を行っています。彼らの研究はメディアに焦点を当てていますが、ランキング・メカニズムとしてのスコアについての彼らのアイデアに便乗して、ユーザーが反応したランキング内の平均的な位置という観点からレコメンダーを評価することができます。 この指標は、レコメンダーが顧客が最終的に購入するものと一致しているかどうかをオフラインで確認するのに役立ちます。
# MAGIC 
# MAGIC しかし、その前に、評価を行うためのレコメンデーションを作成する必要があります。 データセットに含まれるすべてのユーザーに対するレコメンデーションを計算するのは大変なので、ここではランダムに抽出されたユーザーに限定して評価を行います。
# MAGIC 
# MAGIC **注意** 評価対象をごく一部のお客様に限定した場合でも、以下のステップは計算量が多くなります。シャッフル・パーティション・カウントは、使用しているクラスターのサイズに合わせて調整しました。 この方法やその他のパフォーマンスチューニングメカニズムの詳細については、[このドキュメント](https://docs.microsoft.com/en-us/azure/architecture/databricks-monitoring/performance-troubleshooting#common-performance-bottlenecks)を参照してください。

# COMMAND ----------

# DBTITLE 1,Alter Shuffle Partition Count
max_partition_count = sc.defaultParallelism * 100
spark.conf.set('spark.sql.shuffle.partitions', max_partition_count) 

# COMMAND ----------

# DBTITLE 1,Get Similar Users for a Random Sample of Users
# ratio of customers to sample
sample_fraction = 0.10

# calculate max possible distance between users
max_distance = math.sqrt(2)

# calculate min possible similarity (unscaled)
min_score = 1 / (1 + math.sqrt(2))

# remove any old comparisons that might exist
shutil.rmtree('/dbfs/mnt/instacart/gold/similarity_results', ignore_errors=True)

# perform similarity join for sample of users
sample_comparisons = (
  fitted_lsh.approxSimilarityJoin(
    hashed_vectors.sample(withReplacement=False, fraction=sample_fraction), # use a random sample for our target users
    hashed_vectors,
    threshold = max_distance,
    distCol = 'distance'
    )
    .withColumn('similarity', lit(1)/(lit(1)+col('distance')))
    .withColumn('similarity_rescaled', (col('similarity') - lit(min_score)) / lit(1.0 - min_score))
    .selectExpr(
      'datasetA.user_id as user_a',
      'datasetB.user_id as user_b',
      'similarity_rescaled as similarity'
      )
  )

# write output for reuse
(
  sample_comparisons
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/instacart/gold/similarity_results')
  )

display(
  spark.table( 'DELTA.`/mnt/instacart/gold/similarity_results`' )
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 類似性結合を行う際には、ユーザー間の最大可能距離を閾値として設定します。 実際には、LSHごとにすべての類似ユーザーを収集し、以下のステップで固定数の類似ユーザーにフィルタリングできるようにしています。 このようにして、`approxNearestNeighbors()`メソッドコールで行われることを模倣しています。

# COMMAND ----------

# DBTITLE 1,Get k Similar Users
number_of_customers = 10

# get k number of similar users for each sample user
similar_users =  (
    spark.sql('''
      SELECT 
        user_a, 
        user_b, 
        similarity
      FROM (
        SELECT
          user_a,
          user_b,
          similarity,
          ROW_NUMBER() OVER (PARTITION BY user_a ORDER BY similarity DESC) as seq
        FROM DELTA.`/mnt/instacart/gold/similarity_results`
        )
      WHERE seq <= {0}
      '''.format(number_of_customers)
      )
    )

similar_users = similar_users.createOrReplaceTempView('similar_users')
display(similar_users)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 似たようなユーザーのデータセットを使って、前のステップで行ったのと同様の方法で、評価と推奨を組み立てることができます。

# COMMAND ----------

# DBTITLE 1,Retrieve Per-User Product Ratings
similar_ratings = spark.sql('''
    SELECT
      m.user_a,
      m.user_b,
      m.product_id,
      COALESCE(n.normalized_purchases, 0.0) as normalized_purchases,
      m.similarity
    FROM (
      SELECT
        x.user_a,
        x.user_b,
        y.product_id,
        x.similarity
      FROM similar_users x
      CROSS JOIN instacart.products y
      ) m
    LEFT OUTER JOIN ( -- retrieve ratings actually provided by similar users
      SELECT 
        user_id as user_b, 
        product_id, 
        normalized_purchases 
      FROM instacart.user_ratings
      WHERE split = 'calibration'
        ) n
      ON m.user_b=n.user_b AND m.product_id=n.product_id
      ''')

display(similar_ratings)

# COMMAND ----------

# DBTITLE 1,Generate Per-User Recommendations
product_ratings = ( 
   similar_ratings
    .groupBy('user_a','product_id')
      .agg( 
        sum(col('normalized_purchases') * col('similarity')).alias('weighted_rating'),
        sum('similarity').alias('total_weight')
        )
    .withColumn('recommendation_score', col('weighted_rating')/col('total_weight'))
    .select('user_a', 'product_id', 'recommendation_score')
    .orderBy(['user_a','recommendation_score'], ascending=[True,False])
  )

product_ratings.createOrReplaceTempView('product_ratings')

display(
  product_ratings
  )  

# COMMAND ----------

# MAGIC %md
# MAGIC Huらに倣い、推奨度をパーセンタイルランキング（*rank_ui*）に変換し、0.0%を各顧客の最も好ましい推奨度の上位とすることができます。 ここでは、パーセンタイル・ランキングを計算するコードのユニットを、レビューのために別途示します。

# COMMAND ----------

# DBTITLE 1,Convert Recommendations to Percent Ranks
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   user_a as user_id,
# MAGIC   product_id,
# MAGIC   recommendation_score,
# MAGIC   PERCENT_RANK() OVER (PARTITION BY user_a ORDER BY recommendation_score DESC) as rank_ui
# MAGIC FROM product_ratings
# MAGIC ORDER BY user_a, recommendation_score DESC

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC しかし、なぜパーセントランクを使うのでしょうか？ まず、レコメンドスコアをランクに変換することで、スコアを気にすることなく、レコメンドの高いものから低いものへと並べることができます。 最上位の推奨製品の推奨スコアが低くても、その製品が最上位の推奨製品であることに変わりはありません。 第二に、ランクを**パーセントランク**に変換することで、0.0（最もお勧め）から1.0（最もお勧めしない）までの標準的なスケールにランクを置くことができます。 これにより、お客様が推奨製品に対してどのような基準で着地しているかを表す総合評価指標を算出することができます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_rankingseval.gif" width="400">
# MAGIC 
# MAGIC このスケールで考えると、お客様が購入する商品の平均ランクが50％（0.5）程度であれば、私たちの推奨はランダムな提案と変わらないことが予想されます。 平均順位が50％以上であれば、お客様が実際に行っている方向とは反対の方向に提案していることになります。 しかし、平均ランクが50％以下であれば、お客様が認識している好みに沿った方法で製品をお勧めしていることになり、平均値を0.0に近づけるほど、その整合性は向上していきます。
# MAGIC 
# MAGIC そして、校正期間の情報をもとに作成した推奨事項が、評価期間の実際の購入とどのように一致するかを示しています。

# COMMAND ----------

# DBTITLE 1,Evaluate User-Based Recommendations
eval_set = (
  spark
    .sql('''
    SELECT 
      x.user_id,
      x.product_id,
      x.r_t_ui,
      y.rank_ui
    FROM (
      SELECT
        user_id,
        product_id,
        normalized_purchases as r_t_ui
      FROM instacart.user_ratings 
      WHERE split = 'evaluation' -- the test period
        ) x
    INNER JOIN (
      SELECT
        user_a as user_id,
        product_id,
        PERCENT_RANK() OVER (PARTITION BY user_a ORDER BY recommendation_score DESC) as rank_ui
      FROM product_ratings
      ) y
      ON x.user_id=y.user_id AND x.product_id=y.product_id
      ''').cache()
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

# MAGIC %md 
# MAGIC 平均ランキングが低ければ低いほど、お客様は推奨製品リストの上位から製品を購入していることになります。 この指標を別の角度から考えると、スコアが低いほど、ユーザーが欲しいものを手に入れるためにかき分けなければならない*ジャンク*推奨品の数が少ないということになります。ここで返された値は、当社の推奨商品がお客様の購入に非常によくマッチしていることを示しており、これには正当な理由があると考えられます。
# MAGIC 
# MAGIC 評価を作成する際に、ユーザーの暗黙の評価を混ぜることにしました。 つまり、ユーザーが過去に好みを表明した商品は、他の商品よりもおすすめ度が高くなるのです。 お客様が商品を繰り返し購入するパターンが多い食料品店では、お客様が欲しいものを手に入れて取引を完了することが目的である場合は、この方法が有効です。
# MAGIC 
# MAGIC しかし、食料品に限らず、レコメンデーションの手法は常に正しいのでしょうか？ 例えば、クラフトビールのように、目新しさや驚きが評価される商品カテゴリーの場合。過去に購入されたことのある商品を勧めるだけでは、お客様に多様性や魅力に欠けると感じられてしまうかもしれません。 どんなに優れた指標があっても、目標とその評価方法を慎重に検討する必要があります。
# MAGIC 
# MAGIC 
# MAGIC 次のステップに進む前に、私たちの推奨製品を、単に最も人気のある製品をお客様に推奨するという一般的な戦略と比較してみたいと思います。 ここでは、これらの製品の平均パーセントランク指標を計算します。

# COMMAND ----------

# DBTITLE 1,Rank Popular Product Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   product_id,
# MAGIC   PERCENT_RANK() OVER (ORDER BY normalized_purchases DESC) as rank_ui
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.product_id,
# MAGIC     COALESCE(y.normalized_purchases,0.0) as normalized_purchases
# MAGIC   FROM (SELECT product_id FROM instacart.products) x
# MAGIC   LEFT OUTER JOIN instacart.naive_ratings y
# MAGIC     ON x.product_id=y.product_id
# MAGIC   WHERE split = 'calibration'
# MAGIC   )

# COMMAND ----------

# MAGIC %md And using those percent ranks, we can calculate our evaluation metric against the evaluation set:

# COMMAND ----------

# DBTITLE 1,Evaluate Popular Product Recommendations
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   SUM(r_t_ui * rank_ui) / SUM(rank_ui) as mean_percent_rank
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     x.user_id,
# MAGIC     x.product_id,
# MAGIC     x.r_t_ui,
# MAGIC     y.rank_ui
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       p.user_id,
# MAGIC       p.product_id,
# MAGIC       p.normalized_purchases as r_t_ui
# MAGIC     FROM instacart.user_ratings p
# MAGIC     INNER JOIN (SELECT DISTINCT user_a as user_id FROM similar_users) q
# MAGIC       ON p.user_id=q.user_id
# MAGIC     WHERE p.split = 'evaluation' -- the test period
# MAGIC       ) x
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       product_id,
# MAGIC       PERCENT_RANK() OVER (ORDER BY normalized_purchases DESC) as rank_ui
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         x.product_id,
# MAGIC         COALESCE(y.normalized_purchases,0.0) as normalized_purchases
# MAGIC       FROM (SELECT product_id FROM instacart.products) x
# MAGIC       LEFT OUTER JOIN instacart.naive_ratings y
# MAGIC         ON x.product_id=y.product_id
# MAGIC       WHERE split = 'calibration'
# MAGIC       )
# MAGIC     ) y
# MAGIC     ON x.product_id=y.product_id
# MAGIC     )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 最も人気のある製品を推薦した結果は、実に興味深いものでした。 30%の平均パーセントランクを見ると、ランダムに提案するよりも良い結果が出ていますが、ユーザーとその最も似ている仲間を考慮した場合には、良い結果が出ていません。 先ほどのデータから考えられる理由は、ユーザーは私たちが提供する商品の一部を購入する傾向があり、同じ商品を何度も購入するパターンに固定されていることです。 これらの商品がユーザー間であまり重ならない場合（食料品店のシナリオでは、食料品店がこれほど多様な商品を揃える理由は他にありません）、「典型的な」買い物客はほとんどいないと予想され、代わりにユーザーが実際の購入パターンを通して表現する個々の好みを認識したいと考えます。

# COMMAND ----------

# DBTITLE 1,Remove Cached Objects
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
