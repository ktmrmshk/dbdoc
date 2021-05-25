# Databricks notebook source
# MAGIC %md # Chapter 4. アイテムベースのレコメンデーションの構築

# COMMAND ----------

# MAGIC %md このノートブックの目的は、ユーザーベースの協調フィルタの構築に向けて、類似したユーザーを効率的に特定する方法を探ることです。このノートブックは、**Databricks 7.1+ クラスタ**で動作するように設計されています。 

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
# MAGIC ユーザーベースの協調フィルタを構築する際には、製品カタログに掲載されている約5万点の製品すべてに対する暗黙の評価を表すベクトルを、各ユーザーごとに構築する必要があります。 このベクトルをもとに、ユーザー間の類似性を計算します。 約200,000人のユーザーが参加するシステムでは、約200億通りのユーザー比較が発生しますが、これをロケール感度ハッシュ(LSH)を用いてショートカットしました。
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

# DBTITLE 1,(補足, 復習) instacart.user_ratingsテーブルの内容
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

# DBTITLE 1,(1組の)ユーザーペアから評価されているプロダクトの数
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
# MAGIC われわれの製品カタログには約5万点の製品が掲載されており、理論的には12億5000万のユニークな製品ペアを提供することができますが、実際に観測された共起の数（複数の顧客が関与している場合）は5600万点程度になっており、理論的な数の10分の1以下となっています。実際に実績のある製品ペアに焦点を当てることで、関連性のあるものに限定して計算し、問題の複雑さを大幅に軽減することができます。これが、アイテムベースの協調フィルタリングの背景にある[コアインサイト](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)です。
# MAGIC 
# MAGIC しかし、このシナリオでは、具体的にどのように製品を比較すればよいのでしょうか。前回の協調フィルタでは、約5万点の製品それぞれのエントリを含む特徴ベクトルを構築しました。 これを逆にすると、20万人以上のユーザーそれぞれのエントリを含む特徴ベクトルを構築するべきでしょうか。
# MAGIC 
# MAGIC 
# MAGIC 短い答えは、「いいえ」です。 長い答えは、製品の比較は、ある製品ペアを一人のユーザーが両方購入した事実に基づいて評価を実施すると言うことで于S。ユーザーがということです。 その結果、製品ペアに関連付けられた各ユーザーは、評価の両サイドに暗黙の評価を提供します。しかし、ほとんどの製品ペアは、それに関連するユーザーの数が限られています。

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
# MAGIC ある製品の組み合わせを(同時に)購入したユーザーの数(分布)を調べるとの最大が約30,000で、かつ、1つケースしかないと言う点は驚きです。 このような極端なケースを懸念するのであれば、ある製品ペアに紐づくユーザー数が一定の比率を超えた時点で、その製品評価に使用するユーザーをランダムサンプリングして制限するアプローチを使います（このアイデアは、上記のAmazonの論文から直接得られたものです）。しかし、ほとんどの組み合わせは非常に少数のユーザー間でしか発生しないため、各ペアに対して、製品の類似性を測定するための適度なサイズの特徴ベクトルを構築するだけでよいのです。 
# MAGIC 
# MAGIC このことから、ある疑問が生じます。それは、少数のユーザーにしか購入されなかった製品ペア(少数のユーザーしか紐づかない製品ペア)をどう扱うべきか、というものです。2人、3人程度の少数ユーザーのみが購入した製品組み合わせを含めた場合、一般的には考慮されないような製品をレコメンデーションに導入することになるのでしょうか？目的によっては、珍しい商品の組み合わせを含めることは、良いことでもあり、悪いことでもあります。 目新しさや驚きが一般的な目的ではない食料品を扱う場合、共起が少なすぎる商品を除外することは理にかなっていると思われます。後半で、カットオフ値を正確に決定する作業を行いますが、今は、とりあえずここでは製品ベクトルを構築して、notebookを進めましょう。

# COMMAND ----------

# DBTITLE 1,製品の類似性を計算する(Part-1)
def compare_products( data ):
  '''
  引数`data`は以下のカラムのdataframe構成になっている想定:
     => (product_a, product_b, size, values_a, values_b)
  '''
  
  # ベクトルの正規化
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
    values_a = row.values_a # <== 製品の「評価」値はユーザーの「スケール済み購入回数」に基づく(下のSQLを参照)
    
    product_b = row.product_b
    values_b = row.values_b
    
    size = row.size # あとでスパースベクトル化するときに必要
    # -----------------------------------------------------------
    
    # 製品評価値(value_a, b)を正規化(リスケール)させておく
    # -----------------------------------------------------------
    a = Vectors.dense(values_a) 
    a_norm = normalize_vector(a) # <== 全ユーザーからの評価を正規化した値を最終的な製品の評価として使う
    
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


# 比較のため、それぞれの製品のユーザー評価を構成する (user_ratingsをベクトル化したテーブル)
product_comp = (
  spark
    .sql('''
      SELECT
         a.product_id as product_a,
         b.product_id as product_b,
         COUNT(*) as size,
         COLLECT_LIST(a.normalized_purchases) as values_a, -- <== 製品の「評価」値はユーザーの「スケール済み購入回数」に基づく
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

display(product_comp)

# (結果のテーブルから)
# 製品#1と#4136のペアは、3人のユーザーから購入評価されている。
# つまり、この2つの製品を購入したユーザーは3人いる。
# この製品#1と#4136の評価はこれら3人の購入回数に基づく「スケール済みの購入回数」である。
#
# 下のPart-2のセルでは、この結果から、2つのプロダクト間の距離を計算する



# COMMAND ----------

# DBTITLE 1,製品の類似性を計算する(Part-2)
# 製品の類似性を算出
product_sim = (
  product_comp
    .withColumn('id', monotonically_increasing_id()) # <== IDを新たに割り当てる
    .withColumn('subset_id', expr('id % ({0})'.format(sc.defaultParallelism * 10))) # <== 並列処理のためのIDを割り当てる
    .groupBy('subset_id')
    .applyInPandas(
      compare_products, # <== 上記で定義した製品の評価を算出する関数を適用する
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



# COMMAND ----------

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
  .mode('append') # <== 追記モードでDeltaに書き込む
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
      .mode('append') # <== 追記モードでDeltaに書き込む
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
# MAGIC ユーザーベースの協調フィルタでは、類似したユーザーから抽出したユーザー評価の加重平均を計算して、おすすめ商品を生成していました。 
# MAGIC 
# MAGIC ここでは、キャリブレーション期間のデータにおいて、ユーザーが購入した商品を抽出します。これらの製品を軸にして、そのすべての製品ペアを抜き出してきます。 暗黙の評価と類似性スコアは、推薦スコアとして機能する加重平均を構築するために再び使用され、パーセントランクは、推薦を並べるために計算されます。 ここでは、1人のユーザー、`user_id=148`を対象にしてデモを行います。

# COMMAND ----------

# DBTITLE 1,複数回参照されるテーブルをキャッシュしておく
# MAGIC %sql  CACHE TABLE instacart.user_ratings

# COMMAND ----------

# DBTITLE 1,(補足1) 先ほど作った、製品間の距離(=類似度)
# MAGIC %sql
# MAGIC select * from DELTA.`/tmp/mnt/instacart/gold/product_sim`

# COMMAND ----------

# DBTITLE 1,(補足2) ユーザー#148が購入した製品をベースに、それとペアになる製品の類似度を関連づける(join)
# MAGIC %sql
# MAGIC 
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     y.product_a,
# MAGIC     y.product_b as product_id,
# MAGIC     x.normalized_purchases, 
# MAGIC     y.similarity
# MAGIC     --SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score
# MAGIC   FROM instacart.user_ratings x
# MAGIC   INNER JOIN DELTA.`/tmp/mnt/instacart/gold/product_sim` y
# MAGIC     ON x.product_id=y.product_a
# MAGIC   WHERE 
# MAGIC     x.split = 'calibration' AND x.user_id=148 -- <== user #148を抽出

# COMMAND ----------

# DBTITLE 1,ユーザー#148のレコメンデーションを算出
# MAGIC %sql
# MAGIC 
# MAGIC -- 解説: ユーザー#148に対して、製品#55のレコメンデーションスコアの算出を考える
# MAGIC --       前提1: ユーザー#148は製品#55を購入したことがない。よって、この製品#55のユーザー#148による「スケール済み製品の評価」は存在しない。
# MAGIC --       前提2: ユーザー#148が製品#11, #22, #33を購入したとする。よって、このユーザー#148による、この3つの製品の「スケール済み製品の評価」があり、それぞれ0.01, 0.02, 0.03とする。
# MAGIC --       前提3: 製品#55-#11ペア、#55-#11ペア、#55-#33ペアの購入ユーザーが存在する。よって、このペア製品間距離(類似度)があリ、それぞれ0.4, 0.3, 0.2とする。
# MAGIC --       
# MAGIC --       この場合の「ユーザー#148に対して、製品#55のレコメンデーションスコア」は、
# MAGIC --       ユーザーの「スケール済み購入評価」を重みにして、「製品の類似度」の加重平均、として評価する:
# MAGIC --
# MAGIC --     [例]  (0.01*0.4 + 0.02*0.3 + 0.03*0.2) / (0.4+0.3+0.2)
# MAGIC --       
# MAGIC --     意味として、ユーザーが高く評価している製品に関連する(ペアとなる)製品は高く重み付けされ、高く評価される。
# MAGIC --     ユーザーが高く評価した製品Aに「近い」製品Bは、結果としてレコメンデーションスコアが高くなる。
# MAGIC 
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
# MAGIC     SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score -- <== ユーザーの「スケール済み購入評価」を重みにして、「製品の類似度」の加重平均をレコメンデーションスコアとして使う
# MAGIC   FROM instacart.user_ratings x
# MAGIC   INNER JOIN DELTA.`/tmp/mnt/instacart/gold/product_sim` y
# MAGIC     ON x.product_id=y.product_a
# MAGIC   WHERE 
# MAGIC     x.split = 'calibration' AND x.user_id=148 -- <== user #148を抽出
# MAGIC   GROUP BY x.user_id, y.product_b
# MAGIC   )
# MAGIC ORDER BY user_id, rank_ui -- <== レコメンデーションスコアが高い順にソートしている

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC レコメンデーションが構築できました。前回同様にこのレコメンデーションがどの程度「良い」のかをみてみる必要があります。前回のノートブックで採用した平均パーセントランクの評価方法を使って、今回のレコメンデーションを評価してみましょう。 前回と同様に、全ユーザーの10％のサンプルに限定して評価します。

# COMMAND ----------

# DBTITLE 1,ユーザーの10%をランダムにサンプル
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS random_users;
# MAGIC 
# MAGIC -- user_ratingsから10%ランダムにユーザーを選ぶ
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
# MAGIC 
# MAGIC -- テーブルをキャッシュしておく
# MAGIC CACHE TABLE random_users;
# MAGIC 
# MAGIC -- ランダムに選ばれたユーザーを確認する
# MAGIC SELECT * FROM random_users;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(*) FROM random_users;

# COMMAND ----------

_ = spark.sql("CACHE TABLE instacart.user_ratings")
_ = spark.sql("CACHE TABLE DELTA.`/tmp/mnt/instacart/gold/product_sim`")


# COMMAND ----------

# DBTITLE 1,評価メトリックを算出する(制約なし)
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
        normalized_purchases as r_t_ui -- <== 評価・比べるために「スケールされた製品評価」も出しておく
      FROM instacart.user_ratings 
      WHERE split = 'evaluation' -- the test period
        ) m
    INNER JOIN (
      SELECT
        user_id,
        product_id,
        recommendation_score,
        PERCENT_RANK() OVER (PARTITION BY user_id ORDER BY recommendation_score DESC) as rank_ui  -- <== パーセントランク化
      FROM (
        SELECT
          x.user_id,
          y.product_b as product_id,
          SUM(x.normalized_purchases * y.similarity) / SUM(y.similarity) as recommendation_score  -- <==先ほどと同様にレコメンデーションスコアを算出
        FROM instacart.user_ratings x
        INNER JOIN DELTA.`/tmp/mnt/instacart/gold/product_sim` y
          ON x.product_id=y.product_a
        INNER JOIN random_users z   -- <== 上記でランダムに選ばれたユーザーだけに絞る
          ON x.user_id=z.user_id
        WHERE 
          x.split = 'calibration'
        GROUP BY x.user_id, y.product_b
        )
      ) n
      ON m.user_id=n.user_id AND m.product_id=n.product_id
      ''')
  )

# COMMAND ----------

# 結果を出力
display(
  eval_set # <== 上記のクエリ結果
  .withColumn('weighted_r', col('r_t_ui') * col('rank_ui') ) # <= カラムを追加: r_t_ui =「スケール済み製品評価」
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
# MAGIC **なんてことでしょう！** 評価点が悪いだけでなく、ランダムに提案した場合よりも悪くなってしまいました。 *これはどうしたことでしょうか?*
# MAGIC 
# MAGIC 
# MAGIC 最も考えられる理由は、製品ペアを構成する2つの製品を実際に購入したユーザーの数を考慮せずに、すべての製品の組み合わせを利用していることです（本ノートで前述したとおり）。ある組み合わせがごく少数のユーザーから高い評価を受けた場合、その組み合わせがランキングの上位に来るかもしれません。一方で、より人気の高い（つまりより幅広い評価を受けた）製品は下位に押しやられてしまうかもしれません。このようなことを考慮して、製品ペアに関連するユーザー評価数の最小値を設定し、それ以上の評価数を持つ製品のみをレコメンデーション対象にすることにしましょう。
# MAGIC 
# MAGIC また、最小の評価ユーザー数を設けても、レコメンデーション対象の製品数は多いままになります。また、ユーザー数を最小限にしても、推薦する商品の数が多いことに気づくかもしれません。逆に、推薦する商品を(購入)ランキング上位の商品に限定すれば、評価がさらに向上するかもしれません。 このデータセットには、製品の自己推薦が含まれていることを覚えておくことが重要です。*つまり、* 製品Aは製品Aに最も似ているということです。
# MAGIC 
# MAGIC ユーザーの最小値と製品の最大値の調整が、評価指標にどのような影響を与えるかを見てみましょう。
# MAGIC 
# MAGIC 
# MAGIC **注意** このステップは、クラスタに割り当てられたリソースに応じて、実行に時間がかかります。

# COMMAND ----------

# DBTITLE 1,閾値をいくつか設定して、それぞれのレコメンデーションを評価する
_ = spark.sql("CACHE TABLE instacart.user_ratings")
_ = spark.sql("CACHE TABLE DELTA.`/tmp/mnt/instacart/gold/product_sim`")

results = []

# 以下の2種類の閾値を設定してシミュレーションをする
#
# i <= 推奨スコアのランキング上位の閾値。 i=4の場合、レコメンデーションスコアで上位4つの製品のみをレコメンデーションの対象にする
# j <= 製品ペアに紐づくユーザー数の下限閾値。j=3の場合、紐づくユーザーを3人以上を持つ製品ペアをレコメンレーションスコア算出に使用する
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
                WHERE split = 'evaluation' -- "テスト用"期間のデータ
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
                    INNER JOIN DELTA.`/tmp/mnt/instacart/gold/product_sim` y
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
