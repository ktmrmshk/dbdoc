# Databricks notebook source
# MAGIC %md # Chapter 4. アイテムベースのレコメンデーションの構築

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to build and evaluate item-based collaborative filtering recommendations.  This notebook is designed to run on a **Databricks 7.1+ cluster**.

# COMMAND ----------

# DBTITLE 1,必要なライブラリをimport
from pyspark.ml.linalg import Vectors, VectorUDT

from pyspark.sql.functions import col, udf, max, collect_list, lit, monotonically_increasing_id ,expr, coalesce, pow, sum, count
from pyspark.sql.types import *
from pyspark.sql import DataFrame

import pandas as pd
import numpy as np

import math

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: 製品比較データセットの構築
# MAGIC 
# MAGIC ユーザーベースの協調フィルタを構築する際には、製品カタログに掲載されている約5万点の製品すべてに対する暗黙の評価を表すベクトルを、各ユーザーごとに構築する必要があります。 このベクトルをもとに、ユーザー間の類似性を計算します。 約200,000人のユーザーが参加するシステムでは、約200億通りのユーザー比較が発生しますが、これをロケール感度ハッシュを用いてショートカットしました。
# MAGIC 
# MAGIC 
# MAGIC しかし、このアプローチでは、あるユーザーが製品AとBを購入し、さらに製品Aを購入したユーザー、もしくは、製品Bを購入したユーザーが存在する場合にしか推薦ができない(推薦スコアが算出できない)ことになります。このことは、ユーザー間の重なり合う部分に焦点を当てて比較対象を限定するという、ユーザー由来の評価を使用する際の別のアプローチを提供します。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_itembased.gif" width="300">

# COMMAND ----------

# DBTITLE 1,シャッフルパーティションを変更する
_ = spark.conf.set('spark.sql.shuffle.partitions',sc.defaultParallelism * 100)

# COMMAND ----------

# MAGIC %md 
# MAGIC まず、データセットに含まれる製品ペアの数を調べてみましょう。

# COMMAND ----------

# DBTITLE 1,使用するデータをキャッシュしておく
# MAGIC %sql  CACHE TABLE instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,(補足) instacart.user_ratingsテーブルの内容
# MAGIC %sql 
# MAGIC 
# MAGIC select * from instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,(補足) 1組のプロダクトペアを評価しているユーザー数(次のセルで使用するサブクエリ部分)
# MAGIC %sql
# MAGIC SELECT
# MAGIC   a.product_id as product_a,
# MAGIC   b.product_id as product_b,
# MAGIC   COUNT(*) as users
# MAGIC FROM
# MAGIC   instacart.user_ratings a
# MAGIC   INNER JOIN instacart.user_ratings b ON a.user_id = b.user_id
# MAGIC   AND a.split = b.split
# MAGIC WHERE
# MAGIC   a.product_id < b.product_id
# MAGIC   AND a.split = 'calibration'
# MAGIC GROUP BY
# MAGIC   a.product_id,
# MAGIC   b.product_id
# MAGIC HAVING
# MAGIC   COUNT(*) > 1 -- exclude purchase combinations found in association with only one user
# MAGIC   
# MAGIC   
# MAGIC --- 結果から、product#24489, #28204を両方ratingsしているユーザーは2182人存在する

# COMMAND ----------

# DBTITLE 1,1組のユーザーペアから評価されているプロダクトの数
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   COUNT(*) as comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC      a.product_id as product_a,
# MAGIC      b.product_id as product_b,
# MAGIC      COUNT(*) as users
# MAGIC   FROM instacart.user_ratings a
# MAGIC   INNER JOIN instacart.user_ratings b
# MAGIC     ON  a.user_id = b.user_id AND 
# MAGIC         a.split = b.split
# MAGIC   WHERE a.product_id < b.product_id AND
# MAGIC         a.split = 'calibration'
# MAGIC   GROUP BY a.product_id, b.product_id
# MAGIC   HAVING COUNT(*) > 1 -- exclude purchase combinations found in association with only one user
# MAGIC   )    

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC われわれの製品カタログには約5万点の製品が掲載されており、理論的には12億5000万のユニークな製品ペアを提供することができますが、実際に観測された共起の数（複数の顧客が関与している場合）は5600万に近く、理論的な数の10分の1以下となっています。実際に発生している製品ペアに焦点を当てることで、関連性のある可能性のあるものに限定して計算し、問題の複雑さを大幅に軽減することができます。これが、アイテムベースの協調フィルタリングの背景にある[コアインサイト](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)です。
# MAGIC 
# MAGIC しかし、このシナリオでは、具体的にどのように製品を比較すればよいのでしょうか。前回の協調フィルタでは、約5万点の製品それぞれのエントリを含む特徴ベクトルを構築しました。 これを逆にすると、20万人以上のユーザーそれぞれのエントリを含む特徴ベクトルを構築するべきでしょうか。
# MAGIC 
# MAGIC 
# MAGIC 短い答えは、「いいえ」です。 長い答えは、ある製品の比較は、ユーザーがペアの両方の製品を購入しているために行われるということです。 その結果、製品ペアに関連付けられた各ユーザーは、評価の両サイドに暗黙の評価を提供します。しかし、ほとんどの製品ペアは、それに関連する顧客の数が限られています。

# COMMAND ----------

# DBTITLE 1,製品評価ペアに紐づくユーザー数の分布
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT
# MAGIC   users,
# MAGIC   COUNT(*) as occurances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC      a.product_id as product_a,
# MAGIC      b.product_id as product_b,
# MAGIC      COUNT(*) as users
# MAGIC   FROM instacart.user_ratings a
# MAGIC   INNER JOIN instacart.user_ratings b
# MAGIC     ON  a.user_id = b.user_id AND 
# MAGIC         a.split = b.split
# MAGIC   WHERE a.product_id < b.product_id AND
# MAGIC         a.split = 'calibration'
# MAGIC   GROUP BY a.product_id, b.product_id
# MAGIC   HAVING COUNT(*) > 1       -- exclude purchase combinations found in association with only one user
# MAGIC   )
# MAGIC GROUP BY users

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 与えられた製品の組み合わせを購入したことのあるユーザーの最大が30,000未満で、それが一度だけ発生しているのは驚きです。 このような極端なケースを懸念するのであれば、ペアに関連するユーザー数が一定の比率を超えた時点で、考慮するユーザー評価をすべての利用可能な評価のランダムサンプリングに制限することができます（このアイデアは、上記のAmazonの論文から直接得られたものです）。しかし、ほとんどの組み合わせは非常に少数のユーザー間でしか発生しないため、各ペアに対して、製品の類似性を測定するための適度なサイズの特徴ベクトルを構築するだけでよいのです。 
# MAGIC 
# MAGIC このことから、ある疑問が生まれました。それは、少数のユーザーに関連する製品ペアを考慮すべきかということです。 2人、3人、あるいはその他の些細な数のユーザーが購入した組み合わせを含めた場合、一般的には考慮されないような製品をレコメンデーションに導入することになるのでしょうか？目的に応じて、珍しい商品の組み合わせを含めることは、良いことでもあり、悪いことでもあります。 目新しさや驚きが一般的な目的ではない食料品を扱う場合、共起が少なすぎる商品を除外することは理にかなっていると思われます。後半で、カットオフ値を正確に決定する作業を行いますが、今は、とりあえずここでは製品ベクトルを構築して、notebookを進めましょう。

# COMMAND ----------

# DBTITLE 1,製品の類似性を計算する
def compare_products( data ):
  '''
  引数`data`は以下のカラムのdataframe構成になっている想定:
     => (product_a, product_b, size, values_a, values_b)
  '''
  
  def normalize_vector(v):
    norm = Vectors.norm(v, 2)
    ret = v.copy()
    n = v.size
    for i in range(0,n):
      ret[i] /= norm
    return ret
  
  
  # 結果を保持するリストを定義しておく
  results = []
  
  # 引数`data`の行ごとにループ
  for row in data.itertuples(index=False):
    
    # 入力のデータセットをほどく
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
    
    # 距離と類似度を算出
    # -----------------------------------------------------------
    distance = math.sqrt(Vectors.squared_distance(a_norm, b_norm))
    similarity = 1 / (1 + distance)
    similarity_min = 1 / (1 + math.sqrt(2))
    similarity_rescaled = (similarity - similarity_min)/(1 - similarity_min)
   # -----------------------------------------------------------
  
    # 結果を追加
    results += [(
      product_a, 
      product_b, 
      size,
      distance,
      similarity_rescaled
      )]
  
  # 結果をPandas DFとして返す
  return pd.DataFrame(results)


# 比較のため、それぞれの製品のユーザー評価を構成する
product_comp = (
  spark
    .sql('''
      SELECT
         a.product_id as product_a,
         b.product_id as product_b,
         COUNT(*) as size,
         COLLECT_LIST(a.normalized_purchases) as values_a,
         COLLECT_LIST(b.normalized_purchases) as values_b
      FROM instacart.user_ratings a
      INNER JOIN instacart.user_ratings b
        ON  a.user_id = b.user_id AND 
            a.split = b.split
      WHERE a.product_id < b.product_id AND
            a.split = 'calibration'
      GROUP BY a.product_id, b.product_id
      HAVING COUNT(*) > 1
    ''')
  )


# 製品の類似性を算出
product_sim = (
  product_comp
    .withColumn('id', monotonically_increasing_id())
    .withColumn('subset_id', expr('id % ({0})'.format(sc.defaultParallelism * 10)))
    .groupBy('subset_id')
    .applyInPandas(
      compare_products, 
      schema='''
        product_a int,
        product_b int,
        size int,
        distance double,
        similarity double
        '''
      )
  )

# 既存のDeltaデータがある場合は削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/gold/product_sim', ignore_errors=True)

# 製品の類似性をDeltaに書き込む(永続化)
(
  product_sim
  .write
  .format('delta')
  .mode('overwrite')
  .save('/tmp/mnt/instacart/gold/product_sim')  
)

# Deltaに書き込んだ結果を確認する
display(
  spark.table('DELTA.`/tmp/mnt/instacart/gold/product_sim`') 
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC この時点で、私たちの作業は半分ほどしか完了していません。 製品Aと製品Bのペアを構築しましたが、製品Bと製品Aのペアや、製品Aと製品Aのペアは除外しました。(上のクエリのa.product_id < b.product_idの部分がこれにあたります。)  では、これらをデータセットに挿入してみましょう。

# COMMAND ----------

# DBTITLE 1,スキップした製品比較をDeltaに追加する
# flip product A & product B
(
  spark   
  .table('DELTA.`/tmp/mnt/instacart/gold/product_sim`')
  .selectExpr(
    'product_b as product_a',
    'product_a as product_b',
    'size',
    'distance',
    'similarity'
  )
  .write
  .format('delta')
  .mode('append')
  .save('/tmp/mnt/instacart/gold/product_sim')
)

# COMMAND ----------

# DBTITLE 1,製品Aと製品A(同じ製品同士)の類似性も追加(類似度=1.0)
# record entries for product A to product A (sim = 1.0)
(
  spark
    .table('instacart.user_ratings')
    .filter("split='calibration'")
    .groupBy('product_id')
      .agg(count('*').alias('size'))
    .selectExpr(
      'product_id as product_a',
      'product_id as product_b',
      'cast(size as int) as size',
      'cast(0.0 as double) as distance',
      'cast(1.0 as double) as similarity'
      )
    .write
      .format('delta')
      .mode('append')
      .save('/tmp/mnt/instacart/gold/product_sim')
  )

# COMMAND ----------

# DBTITLE 1,キャッシュのリリース
# MAGIC %sql  UNCACHE TABLE instacart.user_ratings

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 問題を見直すことで、比較という課題に対してより直接的なアプローチが可能になりました。 しかし、このデータ構造を使って、具体的にどのようにレコメンデーションを行うのでしょうか？

# COMMAND ----------

# MAGIC %md # Step 2: リコメンデーションを構築する
# MAGIC 
# MAGIC ユーザーベースの協調フィルタでは、類似したユーザーから抽出したユーザー評価の加重平均を計算して、おすすめ商品を生成していました。 ここでは、校正期間中にユーザーが購入した商品を検索します。これらの製品は、製品Aが購入セットの製品の1つであるすべての製品ペアを検索するために使用されます。 暗黙の評価と類似性スコアは、推薦スコアとして機能する加重平均を構築するために再び使用され、パーセントランクは、推薦を並べるために計算されます。 ここでは、1人のユーザー、`user_id=148`を対象にしてデモを行います。

# COMMAND ----------

# DBTITLE 1,複数回参照されるテーブルをキャッシュしておく
# MAGIC %sql  CACHE TABLE instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,ユーザー#148のレコメンデーションを算出
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   user_id,
# MAGIC   product_id,
# MAGIC   recommendation_score,
# MAGIC   PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     y.product_b as product_id,
# MAGIC     SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score
# MAGIC   FROM instacart.user_ratings x
# MAGIC   INNER JOIN DELTA.`/tmp/mnt/instacart/gold/product_sim` y
# MAGIC     ON x.product_id=y.product_a
# MAGIC   WHERE 
# MAGIC     x.split = 'calibration' AND x.user_id=148
# MAGIC   GROUP BY x.user_id, y.product_b
# MAGIC   )
# MAGIC ORDER BY user_id, rank_ui

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 前回同様、推奨事項はありますが、それは良いものでしょうか？前回のノートブックで採用した平均パーセントランクの評価方法を使って、評価データに対して推奨事項を評価してみましょう。 前回と同様に、全ユーザーの10％のサンプルに限定して評価します。

# COMMAND ----------

# DBTITLE 1,ユーザーの10%をランダムにサンプル
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS random_users;
# MAGIC 
# MAGIC CREATE TEMP VIEW random_users 
# MAGIC AS
# MAGIC   SELECT user_id
# MAGIC   FROM (
# MAGIC     SELECT DISTINCT 
# MAGIC       user_id
# MAGIC     FROM instacart.user_ratings
# MAGIC     ) 
# MAGIC   WHERE rand() <= 0.10;
# MAGIC   
# MAGIC CACHE TABLE random_users;
# MAGIC 
# MAGIC SELECT * FROM random_users;

# COMMAND ----------

# DBTITLE 1,Calculate Evaluation Metric without Constraints
eval_set = (
  spark
    .sql('''
    SELECT 
      m.user_id,
      m.product_id,
      m.r_t_ui,
      n.rank_ui
    FROM (
      SELECT
        user_id,
        product_id,
        normalized_purchases as r_t_ui
      FROM instacart.user_ratings 
      WHERE split = 'evaluation' -- the test period
        ) m
    INNER JOIN (
      SELECT
        user_id,
        product_id,
        recommendation_score,
        PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
      FROM (
        SELECT
          x.user_id,
          y.product_b as product_id,
          SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score
        FROM instacart.user_ratings x
        INNER JOIN DELTA.`/mnt/instacart/gold/product_sim` y
          ON x.product_id=y.product_a
        INNER JOIN random_users z
          ON x.user_id=z.user_id
        WHERE 
          x.split = 'calibration'
        GROUP BY x.user_id, y.product_b
        )
      ) n
      ON m.user_id=n.user_id AND m.product_id=n.product_id
      ''')
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
# MAGIC 
# MAGIC **すごい！** 評価点が悪いだけでなく、ランダムに提案した場合よりも悪くなってしまいました。 *これはどうしたことでしょうか?*
# MAGIC 
# MAGIC 
# MAGIC 最も考えられる理由は、製品ペアを構成する2つの製品を実際に購入したユーザーの数を考慮せずに、すべての製品の組み合わせを利用していることです（本ノートで前述したとおり）。ある組み合わせがごく少数のユーザーから高い評価を受けた場合、その組み合わせがランキングの上位に来るかもしれませんが、より人気の高い（つまりより幅広い評価を受けた）製品は下位に押しやられてしまうかもしれません。 このようなことを考慮して、私たちは、製品ペアに関連する最小数のユーザー評価を持つ製品に推奨製品を限定することがあります。
# MAGIC 
# MAGIC また、ユーザー数を最小限に抑えても、推奨する製品数が多くなる可能性があることも認識しています。 また、ユーザー数を最小限にしても、推薦する商品の数が多いことに気づくかもしれません。もし、推薦する商品をランキング上位の商品に限定すれば、評価がさらに向上するかもしれません。 このデータセットには、製品の自己推薦が含まれていることを覚えておくことが重要です。*つまり、*製品Aは製品Aに最も似ているということです。
# MAGIC 
# MAGIC ユーザーの最小値と製品の最大値の調整が、評価指標にどのような影響を与えるかを見てみましょう。
# MAGIC 
# MAGIC 
# MAGIC **注意** このステップは、クラスタに割り当てられたリソースに応じて、実行に時間がかかります。

# COMMAND ----------

# DBTITLE 1,Iterate over Thresholds
_ = spark.sql("CACHE TABLE instacart.user_ratings")
_ = spark.sql("CACHE TABLE DELTA.`/tmp/mnt/instacart/gold/product_sim`")

results = []

for i in range(1,21,1):
  print('Starting size = {0}'.format(i))
  
  for j in [2,3,5,7,10]:
  
    rank = (
        spark
          .sql('''
            SELECT
              SUM(r_t_ui * rank_ui) / SUM(r_t_ui) as mean_percent_rank
            FROM (
              SELECT 
                m.user_id,
                m.product_id,
                m.r_t_ui,
                n.rank_ui
              FROM (
                SELECT
                  user_id,
                  product_id,
                  normalized_purchases as r_t_ui
                FROM instacart.user_ratings 
                WHERE split = 'evaluation' -- the test period
                  ) m
              INNER JOIN (
                SELECT
                  user_id,
                  product_id,
                  recommendation_score,
                  PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui
                FROM (
                  SELECT
                    user_id,
                    product_id,
                    SUM(normalized_purchases * similarity) / SUM(similarity) as recommendation_score
                  FROM (
                    SELECT
                      x.user_id,
                      y.product_b as product_id,
                      x.normalized_purchases,
                      y.similarity,
                      RANK() OVER (PARTITION BY x.user_id, x.product_id ORDER BY y.similarity DESC) as product_rank
                    FROM instacart.user_ratings x
                    INNER JOIN DELTA.`/mnt/instacart/gold/product_sim` y
                      ON x.product_id=y.product_a
                    LEFT SEMI JOIN random_users z
                      ON x.user_id=z.user_id
                    WHERE 
                      x.split = 'calibration' AND
                      y.size >= {0}
                    )
                  WHERE product_rank <= {1}
                  GROUP BY user_id, product_id
                  )
                ) n
                ON m.user_id=n.user_id AND m.product_id=n.product_id
              )
            '''.format(i,j)
              ).collect()[0]['mean_percent_rank']
        )

    results += [(i, j, rank)]

  
display(
  spark
    .createDataFrame(results, schema="min_users int, max_products int, mean_percent_rank double")
    .orderBy('min_users','max_products')
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC このように、推奨事項に少し制約を加えることで、より良い結果が得られるようです。 ETLサイクルを制限し、クエリのパフォーマンスを向上させるために、これらの制約を製品比較データセットの構築に移すことを検討しています。 ユーザーベースの協調フィルタで見られたようなレベルのパフォーマンスではありませんが、結果は悪くありません（さらにロジックを調整すれば改善されるかもしれません）。

# COMMAND ----------

# DBTITLE 1,Clear Cached Data
# MAGIC %sql  
# MAGIC UNCACHE TABLE instacart.user_ratings;
# MAGIC UNCACHE TABLE random_users;
# MAGIC UNCACHE TABLE DELTA.`/tmp/mnt/instacart/gold/product_sim`;
