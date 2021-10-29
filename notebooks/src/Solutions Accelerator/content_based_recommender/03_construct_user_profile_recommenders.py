# Databricks notebook source
# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC このノートブックの目的は、コンテンツベースの推奨を行う際に、ユーザーの評価をどのように考慮するかを検討することです。このノートブックは **Databricks ML 7.3+ クラスタ** で実行する必要があります。
# MAGIC 
# MAGIC これまで私たちは、コンテンツベースのフィルターを使って、製品の特徴の類似性を利用して類似アイテムを特定してきました。 しかし、ユーザーからのフィードバック（明示的、暗黙的を問わず）を利用することで、お客様が好きそうな商品のプロファイルを構築し、より幅広く、時には多彩な商品を配置することができるようになります。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_profile_recs2.png" width=600>

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
from pyspark.sql.functions import col, expr, sum, count
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import Normalizer
import mlflow.spark

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: プロダクトプロファイルの取得
# MAGIC 
# MAGIC これまでは、商品の特徴だけをもとにコンテンツベースの推奨を行ってきました。これは、お客様が製品の代替品を検討することを目的としている場合に、検討中の製品に類似したアイテムを特定するための手段となります。
# MAGIC 
# MAGIC しかし、今回のレコメンダーは少し違います。商品の特徴の好みから、お客様の共感を得られそうな特徴を学びます。 これにより、より多様な製品をお勧めすることができますが、お客様にアピールする可能性のある機能を考慮してお勧めすることができます。
# MAGIC 
# MAGIC このレコメンダーを使い始めるために、Word2Vecの特徴を、前のノートブックで生成されたクラスタ/バケットの割り当てとともに取得します。

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

# MAGIC %md # Step 2: ユーザープロファイルを構成する
# MAGIC 
# MAGIC 次のステップでは、お客様の明示的な製品レビューに基づいて、各お客様が好む機能の加重平均を構築します。 このレコメンダーを評価するために、レビューをキャリブレーションセットとホールドアウトセットに分けます。 ホールドアウトセットは、ユーザーが提供した最後の2つの評価で構成され、キャリブレーションセットは、それ以前にユーザーが生成したすべての評価で構成されます。

# COMMAND ----------

# DBTITLE 1,レビューをキャリブレーションとホールドアウトのセットに分ける
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

# MAGIC %md これで、個々のユーザーが評価した製品の特徴の平均値を、その評価をもとにして組み立てることができます。 その際に注意しなければならないのは、ここでは1～5の評価をしていることです。 ほとんどのユーザーは、Amazonのようなサイトで購入した商品の大部分は評価されません。 一般的なユーザーは、製品に非常に満足した場合や非常に不満な場合に評価をするため、評価はスケールの両端に少し偏っていることが予想されます。

# COMMAND ----------

display(
  reviews_cali
    .groupBy('rating')
      .agg( count('*').alias('instances'))
    .orderBy('rating')
  )

# COMMAND ----------

# MAGIC %md 尺度の中間にある評価を見逃す可能性があることに加えて、評価の意味を考える必要があります。 それは好みを表しているのか、それとも製品やサプライヤーに対する反応なのか。お客様がその製品に興味を持って購入したのであれば、それはおそらく好みのより正確な指標となるでしょう。 その商品に失望したということは、お客様の関心事に合致した期待に応えられなかったということでしょう。しかし、3はどうでしょうか？ 3が好みの表現のベースラインであるならば、3以下の評価は平均値をマイナス方向に導くために使用すべきでしょうか？
# MAGIC 
# MAGIC これらの懸念にどのように対処するかは、評価を取り巻く特定のビジネス状況に大きく依存します。ここでは、ユーザープロファイルの作成を、お客様が4と5の評価を使って強い好みを表明した場合のみに限定し、その他の評価はすべて無視します。 そして、4と5の評価はそのままにしておきます。 繰り返しになりますが、これがお客様のビジネスにとって正しい選択であるとは限りません。
# MAGIC 
# MAGIC 特徴ベクトルに対する加重平均を行うには、2つの選択肢があります。 カスタム関数を作成してベクターに適用するか、[Summarizer transformer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.stat.Summarizer)を使用するかです。Summarizer トランスフォーマーを使用すると、わずかなコードでベクトルに対する単純な集約を実行でき、加重平均のサポートも含まれています。 これは、平均化された後に正規化を適用する必要があることを意味しています。
# MAGIC 
# MAGIC **注** 必要に応じて、特徴量の最大値を取るなど、他のアプローチも使用できます。

# COMMAND ----------

# DBTITLE 1,ユーザープロファイルを構成する
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

# MAGIC %md これで、各ユーザーの特徴ベクトルは、そのユーザーの重み付けされた特徴の好みを表しています。 このベクトルは、各ユーザーにとって（そのユーザーのフィードバックに基づく）「理想の製品」を表していると考えることができます。 私たちの目標は、この「理想」に近い製品を見つけることです。そのためには、各特徴ベクトルをクラスター／バケットに割り当てる必要があります。バケット化されたプロファイルは、後で使用するために保存されます。
# MAGIC 
# MAGIC **注意** 以前のノートブックに保存されているクラスタリングモデルを再利用しています。このモデルをmlflowレジストリから取得すると、冗長な警告メッセージが表示されますが、無視してかまいません。
# MAGIC 
# MAGIC **注意** このデータセットをストレージに書き込む際、いくつかの問題が発生しました。 まだ調査中ですが、今のところParquetでデータを永続化しています。

# COMMAND ----------

# DBTITLE 1,クラスター/バケットの割り当てとユーザープロファイルの永続化
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

# DBTITLE 1,バケット別の分布を調べる
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as profiles
# MAGIC FROM PARQUET.`/mnt/reviews/gold/user_profiles_cali`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# MAGIC %md # Step 3: レコメンデーションの構築と評価
# MAGIC 
# MAGIC これで、様々な機能を持つ製品と、製品の機能に関する好みを表すユーザープロファイルが揃いました。 おすすめしたい製品を見つけるために、製品とこれらのユーザーの好みの類似性を計算します。
# MAGIC 
# MAGIC これらの推薦を評価するために、ユーザーを少数のランダムなサンプルに限定し、協調フィルタで行ったのと同様に、加重平均パーセントスコアを計算します。この評価方法の詳細については、関連するノートブックを参照してください。

# COMMAND ----------

# DBTITLE 1,距離計算の関数を定義する
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

# DBTITLE 1,レビュアーのランダムなサンプルを取る
cali_profiles= spark.table('PARQUET.`/mnt/reviews/gold/user_profiles_cali`').sample(False, 0.01)

cali_profiles.count()

# COMMAND ----------

# DBTITLE 1,選択されたユーザーへの推奨事項の決定
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

# DBTITLE 1,一人のレビュアーの結果を選択して表示
# we are retrieving a subset of recommendations for one user so that the range of rank_ui values is more visible
display(
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`')
    .join( spark.table('DELTA.`/mnt/reviews/gold/user_profile_recommendations`').limit(1), on='reviewerID', how='left_semi' )
    .sample(False,0.01) 
    .orderBy('rank_ui', ascending=True)
  )

# COMMAND ----------

# MAGIC %md そして、お客様が次の買い物をする場所が、レコメンデーションの中でわかるようになりました。

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

# MAGIC %md 協調フィルタと比較して、ここでの平均パーセントランクスコアはそれほど高くありません。多くの製品が混在しているため、多様な製品を購入するお客様は、さまざまな好みを持っている可能性があり、それらが1つのユーザープロファイルに集約されているのです。 例えば、私はある製品カテゴリーでは目新しさや高品質を好みますが、他のカテゴリーでは低価格や一貫性・信頼性を全く好みません。私のような人には、特定の製品カテゴリーに合わせて、異なるプロファイルを構築する方が理にかなっているかもしれません。しかし、コンテンツベースのレコメンダーにレーティングを適用するための基本的な技術は、明確であり、これらのシナリオや他のシナリオに適応できるものでなければなりません。
