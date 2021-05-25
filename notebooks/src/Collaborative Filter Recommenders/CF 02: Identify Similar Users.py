# Databricks notebook source
# MAGIC %md 
# MAGIC # Chapter2: 類似ユーザーを特定する

# COMMAND ----------

# MAGIC %md 
# MAGIC このノートブックの目的は、ユーザーベースの協調フィルタの構築に向けて、類似したユーザーを効率的に特定する方法を探ることです。このノートブックは、**Databricks 7.1+ クラスタ**で動作するように設計されています。 

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC 協調フィルタは，現代のレコメンデーション体験を実現する重要な要素です．「***あなたに似たお客様はこんなものも買っています***」というタイプのレコメンデーションは、興味を引く可能性の高い商品を特定し、購入商品の幅を広げたり、お客様のショッピングカートをより早く多くの商品で満たしてもらうための重要な手段となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">
# MAGIC 
# MAGIC レコメンダーを構築するために必要なトランザクションデータが揃ったところで、このタイプのフィルターの背後にあるメカニズムに目を向けてみましょう。製品購入のパターンが似ている顧客を特定することが問題の中心となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_userbasedcollab.gif" width="300">

# COMMAND ----------

# DBTITLE 1,必要なライブラリのimport
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH

from pyspark.sql.functions import col, udf, max, collect_list, lit, monotonically_increasing_id ,expr, coalesce, pow, sum
from pyspark.sql.types import *
from pyspark.sql import DataFrame

import pandas as pd
import numpy as np

import math

# COMMAND ----------

# MAGIC %md # Step 1:  評価データセット(ratings dataset)の探索
# MAGIC 
# MAGIC ユーザーベースの協調フィルタ（CF）の生成に入る前に、まずデータセットの概要を把握するために、レコメンデーションの対象になる顧客の数を考えてみましょう。

# COMMAND ----------

# DBTITLE 1,レコメンドの対象の顧客数(ユーザー数)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 'users' as entity, COUNT(DISTINCT user_id) as instances FROM instacart.user_ratings

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ユーザーベースのレコメンデーションを構築するためには、約20万人のユーザーをそれぞれ比較する必要があります。単純に考えると、これは約400億回の比較に相当します(`=200,000 * 200,000`)。 
# MAGIC 
# MAGIC 正確には、ユーザーAとユーザーBの比較は、当然ユーザーBとユーザーAの比較と同じ結果になるので、この回数は実質半分になります。 また、ユーザーAと自分との比較が一定の結果になるため、必要な比較回数を減らすことができます。ただし、それらは全体から見ると微々たるもので、結局は約200億回の比較が必要になります。 
# MAGIC 
# MAGIC ユーザーペア(2人のユーザーの組み合わせ、つまり、ユーザーAとユーザーBを1組のユーザーペアと呼びます)ごとに、各製品に関連する暗黙の評価を比較する必要があります。 (次のクエリ結果から、)このデータセットでは、各ユーザーペアに対して、約50,000件の製品レベルの比較を行う必要があります。

# COMMAND ----------

# DBTITLE 1,データセットにある製品数
# MAGIC %sql 
# MAGIC 
# MAGIC SELECT 'products', COUNT(DISTINCT product_id) FROM instacart.products

# COMMAND ----------

# MAGIC %md
# MAGIC しかし、ほとんどのユーザー(お客様)は、わずかな種類の製品しか購入しません。 ユーザーと製品の関連性は100億通りあると言われていますが、次のクエリ結果から、今回のデータセットでは約1400万通りしかないことがわかります。

# COMMAND ----------

# DBTITLE 1,「製品数」と「(それを購入したことのある)ユーザー」のペア数
# MAGIC %sql
# MAGIC 
# MAGIC -- 前のnotebookで算出した"user_ratings"のusr_id, product_idのユニークレコード数
# MAGIC 
# MAGIC SELECT 'ratings', COUNT(*) as instances
# MAGIC FROM (
# MAGIC   SELECT DISTINCT user_id, product_id 
# MAGIC   FROM instacart.user_ratings
# MAGIC   );

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 不必要なオーバーヘッドを避けるために、ユーザー間の比較をどのように効率的に行うかを考える必要があるということです。

# COMMAND ----------

# MAGIC %md # Step 2: 似たようなユーザーを特定する (Brute-Force(単純総当たり)の手法)
# MAGIC 
# MAGIC ユーザーベースの協調フィルタは、類似したユーザーの評価から構築された加重平均に基づいています。このようなレコメンダーを構築するための出発点は、ユーザー間の比較を構築することです。
# MAGIC 
# MAGIC 単純な方法から試してみましょう。*ブルートフォース(総当たり)* では、各顧客を他の顧客と比較します。この作業には、ユーザー同士の組み合わせをクラスターに分散させることで、Databricksのプラットフォームをフルに活用することができます。
# MAGIC 
# MAGIC **注意** ここでは、デモ用をタイムリーに実行できるようにするため、LIMIT句を使用して、比較対象を100人のユーザーのサブセットのみに限定しています。

# COMMAND ----------

# MAGIC %sql
# MAGIC         SELECT
# MAGIC           user_id,
# MAGIC           product_id,
# MAGIC           normalized_purchases
# MAGIC         FROM instacart.user_ratings
# MAGIC         WHERE 
# MAGIC           split = 'calibration' AND
# MAGIC           user_id IN (  -- ユーザー数を絞る
# MAGIC             SELECT DISTINCT user_id 
# MAGIC             FROM instacart.user_ratings 
# MAGIC             WHERE split = 'calibration'
# MAGIC             ORDER BY user_id 
# MAGIC             LIMIT 100   
# MAGIC             )

# COMMAND ----------

# DBTITLE 1,ユーザーAとユーザーBの比較を組み立てる (Part-1)
# 解説: 前のnotebookで用意した"user_rating"は、ユーザー毎に、複数の製品(product_id)の
#     評価値(実際は購入回数から算出されている)の行がある。これだと、ユーザー毎の比較がやりづらいので、
#     ここでは、ユーザー毎に、そのユーザーが持っている製品評価を一行にまとめる処理を行う(Part-1部分)。
#     その後、ユーザー間の比較を実施する(Part-2部分)。

ratings = (
  spark
    .sql('''
      SELECT
        user_id,
        COLLECT_LIST(product_id) as products_list, -- <== COLLECT_LIST()で、GroupByした結果を配列(リスト化)としてまとめる。
        COLLECT_LIST(normalized_purchases) as ratings_list -- <== 同上
      FROM (
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE 
          split = 'calibration' AND
          user_id IN (  -- ユーザー数を絞る <== "user_rating"の中から、100人に絞る
            SELECT DISTINCT user_id 
            FROM instacart.user_ratings 
            WHERE split = 'calibration'
            ORDER BY user_id 
            LIMIT 100          -- limit should be > 1
            )
        )
      GROUP BY user_id
      ''')
  ).cache() # この後何回か参照されるdataframeなので、キャッシュしておく。

display(ratings)

# COMMAND ----------

# DBTITLE 1,ユーザーAとユーザーBの比較を組み立てる (Part-2)
# 解説: ひと組のユーザーペアでクロス分析(実施つcross join)するため、
#      上記で整理した"rating"テーブルを2つ作り、user_idが異なる行同士をjoinさせる


# User-Aについて構築(1つ目のテーブル)
ratings_a = (
            ratings
              .withColumnRenamed('user_id', 'user_a')
              .withColumnRenamed('products_list', 'indices_a')
              .withColumnRenamed('ratings_list', 'values_a')
            )

# User-Bについて構築(2つ目のテーブル)
ratings_b = (
            ratings
              .withColumnRenamed('user_id', 'user_b')
              .withColumnRenamed('products_list', 'indices_b')
              .withColumnRenamed('ratings_list', 'values_b')
            )

# 商品IDを保持するために必要な要素数を算出（インデックス0を考慮して+1しておく)
# このサイズはこの後使用する、スパースベクタ化で必要。
size = spark.sql('''
  SELECT 1 + COUNT(DISTINCT product_id) as size 
  FROM instacart.products
  ''')

# User-AとUser-Bを関連づける(Join)
a_to_b = (
  ratings_a
  .join(ratings_b, [ratings_a.user_a < ratings_b.user_b]) # 対称性から必要な部分だけにする  
).crossJoin(size)

display(a_to_b)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ユーザー間のマッチと比較データが整理されたところで、効率的な比較操作を可能にするために、データをスパース・ベクトル形式に変換してみましょう。

# COMMAND ----------

# DBTITLE 1,比較するための関数を定義
def compare_users( data ):
  '''
  引数の`data`は以下を行に持つdataframe:
     user_a, indices_a, values_a, user_b, indices_b, values_b, size
  '''
  
  # 結果を記録しておくリストを用意
  results = []
  
  # サイズを抽出しておく
  size = data['size'][0]
  
  # `data`の各データに関して、foreach loopを回す
  for row in data.itertuples(index=False):
    
    # それぞれのデータを抽出しておく
    # -----------------------------------------------------------
    user_a = row.user_a
    indices_a = row.indices_a
    values_a = row.values_a
    
    user_b = row.user_b
    indices_b = row.indices_b
    values_b = row.values_b
    # -----------------------------------------------------------
    
    # ユーザー間比較のためにデータを再構成
    # -----------------------------------------------------------
    # User-Aの評価値(ratings)をスパースベクターとして構成する
    ind_a, val_a = zip(*sorted(zip(indices_a, values_a)))
    a = Vectors.sparse(size, ind_a, val_a)

    # User-Bの評価値(ratings)をスパースベクターとして構成する
    ind_b, val_b = zip(*sorted(zip(indices_b, values_b)))
    b = Vectors.sparse(size, ind_b, val_b)
    # -----------------------------------------------------------
    
    # User-A, B間のユークリッド距離を計算する
    # -----------------------------------------------------------
    distance = math.sqrt(Vectors.squared_distance(a, b))
   # -----------------------------------------------------------
  
    # 結果を記録しておく
    results += [(
      user_a, 
      user_b, 
      distance
      )]
  
  # 最終結果を返す
  return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ここで定義した関数は、`data`というpandasのデータフレームを受け取ることになっています。 このデータフレームにはレコードが含まれており、各レコードは1人のユーザー(ユーザーA)ともう1人のユーザー(ユーザーB)との間の比較を表しています。また、各ユーザーには、共同でソートされた製品IDと評価のリストが提供されます。この関数は、これらの評価を使用して、比較に必要なスパース ベクトルを組み立てる必要があります。 その後、単純なユークリッド距離計算が実行され、製品評価の観点からユーザー間の距離が *どの程度離れているか* が判断されます。
# MAGIC 
# MAGIC この作業を行う関数は、[pandas UDF using the Spark 3.0 syntax](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#pandas-function-apis)と定義されています。 この関数をSparkのデータフレームで整理されたデータに適用します。 Sparkデータフレームを使うことで、データをクラスタのワーカーノードに分散させ、データセットのサブセットに関数を並列に適用することができます。
# MAGIC 
# MAGIC 
# MAGIC データがクラスタ全体に均等に分散されるように（つまり、各ワーカーに適度に一貫した作業量が割り当てられるように）、データセットの各レコードのidを計算し、モジュロ計算を実行してデータのサブセットを分散セットに割り当てます。`monotonically_increasing_id()`関数は、完全に連続したid値を生成するわけではありませんが、モジュロ演算によって、同じような数のレコードを含むセットに到達することが期待できます。 このステップでは、id を計算してそれに対するデータのサブセットを作成するだけでなく、データをサブセットにグループ化してシャッフル操作を行うというオーバーヘッドがあります。小さなデータセットの場合、このステップは大きなオーバーヘッドをもたらす可能性があるため、このようなアクションを実装する前に、各データフレーム・パーティションのレコード数を慎重に評価する必要があります。
# MAGIC 
# MAGIC **注意** ここでの4,950件のユーザー比較は、そこそこのサイズのワーカーでもメモリ不足エラーを起こすほどの大きさではありません。 行数の上限は、誰かが上記で課したLIMITを引き上げ、実行される比較の数を指数関数的に増加させた場合に備えたものです。

# COMMAND ----------

# DBTITLE 1,ユーザーの相似性を算出する (100 users)
# redistribute the user-to-user data and calculate similarities
similarities = (
  a_to_b
    .withColumn('id', monotonically_increasing_id())
    .withColumn('subset_id', expr('id % ({0})'.format(sc.defaultParallelism * 10)))
    .groupBy('subset_id')
    .applyInPandas(
      compare_users, # <== ここで先ほど定義したユーザー間の距離を算出する関数をDataframe全体に適用している。
      schema='''
        user_a int, 
        user_b int, 
        distance double
        ''')
    )

# force full execution cycle (for timings)
similarities.count()

# COMMAND ----------

# DBTITLE 1,上記で算出したユーザー間の距離(similarities)を確認する
display(
  similarities
)

# COMMAND ----------

# DBTITLE 1,キャッシュデータをクリア
# unpersist dataset that is no longer needed
_ = ratings.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC ブルートフォース比較のタイミングを考える前に、上の類似性データセットの user_a と user_b の間の距離をよく見てください。 L2正規化された集合では、2点間の最大距離は2の平方根になるはずです。
# MAGIC 
# MAGIC 注: L2正規化によって全ての点が半径1の単位円/球/超球上の第一象限にある。このため距離が最大になるのは、`root(2)=1.414`になる。
# MAGIC 
# MAGIC 多くのユーザーがこの距離に近い距離で離れているという事実は、(1)多くのユーザーが互いに本当に似ていないこと、(2)比較的似ているユーザーでも、多くの製品機能の違いの累積効果により、かなりの距離で離れてしまうこと、を反映しています。
# MAGIC 
# MAGIC 
# MAGIC この後の問題は、しばしば「次元の呪い」と呼ばれます。 類似性の計算に使用する製品を、かなり人気のある製品のサブセットに限定したり、このデータに対して次元削減技術（主成分分析など）を適用することで、この影響を軽減できるかもしれません。ここではこれらのアプローチのデモは行いませんが、実際の実装では検討したいことです。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ブルートフォース比較のパフォーマンスに話を戻すと、ユーザー数が100人の場合、処理時間はわずか数秒です。 ワーカーノードの数を固定したままユーザー数を調整すると、時間が指数関数的に増加することがわかります。 以下は、ワーカー1人あたり4vCPUの4ワーカーノードクラスタを使用した場合の、ユーザー数に対する比較時間のグラフです。 ユーザー数に対して計算時間が指数関数的に増加していることがわかります。
# MAGIC 
# MAGIC **注意** このようなデータを視覚化するには散布図が一般的ですが、棒グラフを使うと、より明確にポイントを把握することができます。

# COMMAND ----------

# DBTITLE 1,ユーザー数を変化させた場合のユーザ間比較にかかる秒数
timings = spark.createDataFrame(
  [
  (10,2.13),
  (10,2.31),
  (10,2.05),
  (10,1.77),
  (10,1.92),
  (100,2.62),
  (100,2.32),
  (100,2.12),
  (100,2.22),
  (100,2.32),
  (1000,138.0),
  (1000,148.0),
  (1000,150.0),
  (1000,148.0),
  (1000,151.0),
  (10000,13284.0),
  (10000,11808.0),
  (10000,12168.0),
  (10000,12392.0)
  ],
  schema='users int, seconds float'
  )

display(
  timings
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ユーザー比較のためのブルートフォース（総当り）方式では、計算が完了するまでかなりの時間を待つか、時間を安定させるために指数関数的に近い数のリソースを追加する必要があります。 これらのアプローチはどちらも持続可能ではありません。そのため、比較の実行を制限する方法を見つける必要があるのです。

# COMMAND ----------

# MAGIC %md # Step 3: 似たようなユーザーを特定する (LSH手法)
# MAGIC 
# MAGIC ブルートフォース比較の代わりに、[*locality-sensitive hashing* (LSH)](https://www.slaney.org/malcolm/yahoo/Slaney2008-LSHTutorial.pdf)と呼ばれる技術を使って、ユーザーを **潜在的に** 類似したユーザーのバケットに素早く分割することができます。LSHの基本的な考え方は、ユーザーをバケットに分け、共有されたバケット内のユーザーだけに比較を限定することで、比較する必要のあるユーザーの数を制限できるというものです。 ユーザーをどのバケットに入れるかは、ランダムな超平面（この2D画像では線で表されている）を生成し、メンバーがその平面の上にいるか下にいるかを判断することで決定します。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash00x.png' width='500'>
# MAGIC 
# MAGIC 複数の超平面を生成することで、ユーザーを適度な大きさのバケットに分けます。 あるバケットのユーザーは、他のバケットのメンバーよりも互いに似ていると予想されます。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash01x.png' width='500'>
# MAGIC 
# MAGIC このプロセスは、すべてのメンバーが他のバケットのメンバーよりも互いに似ているバケットにユーザーを完全に分けることを保証するものではありませんが、ユーザーを*似ている*グループに分けるための迅速な方法を提供します。
# MAGIC 
# MAGIC Sparkに実装されているLSHを利用するには、ユーザーの商品評価をスパースなベクトルにまとめる必要があります。このようなベクトルを生成するために、`to_vector()`というユーザー定義の関数を作成します。 この関数には，ベクトルのインデックスとなる商品IDのリストだけでなく，それらの商品に対する評価のリストも渡す必要があります． また、この関数は、いくつのインデックスポジションがあるのかを知る必要があります。 整数の製品IDをインデックス値として使用するので、すべての製品に対して十分な数のインデックスポジションを確保するために、データセット内の最大製品ID値+1を`to_vector()` UDFに通知します。

# COMMAND ----------

# DBTITLE 1,評価値をスパースベクタに変換する関数を定義
@udf(VectorUDT())
def to_vector(size, index_list, value_list):
    
    # indexで各ペアをsortしておく
    ind, val = zip(*sorted(zip(index_list, value_list)))
    
    # sparseベクタ形式にして返す
    return Vectors.sparse(size, ind, val)

# SQL内で使用できるように関数登録しておく
_ = spark.udf.register('to_vector', to_vector)

# COMMAND ----------

# MAGIC %md
# MAGIC さて、データセットを準備しましょう。 ここでは、同じロジックを複数の方法で実装できるように、最初にSQLを、次にPythonを使ってみます。ここでは、20万人以上のユーザーが含まれる全ユーザーデータセットを使用していることに注意してください。

# COMMAND ----------

# DBTITLE 1,ユーザーベクターの準備 (by SQL)
# MAGIC %sql
# MAGIC 
# MAGIC SELECT  -- convert lists into vectors
# MAGIC   user_id,
# MAGIC   to_vector(size, index_list, value_list) as ratings
# MAGIC FROM ( -- aggregate product IDs and ratings into lists
# MAGIC   SELECT
# MAGIC     user_id,
# MAGIC     (SELECT MAX(product_id) FROM instacart.products) + 1 as size,
# MAGIC     COLLECT_LIST(product_id) as index_list,
# MAGIC     COLLECT_LIST(normalized_purchases) as value_list
# MAGIC   FROM ( -- all users, ratings
# MAGIC     SELECT
# MAGIC       user_id,
# MAGIC       product_id,
# MAGIC       normalized_purchases
# MAGIC     FROM instacart.user_ratings
# MAGIC     WHERE split = 'calibration'
# MAGIC     )
# MAGIC   GROUP BY user_id
# MAGIC   )

# COMMAND ----------

# DBTITLE 1,ユーザーベクターの準備 (by Python)
# assemble user ratings
user_ratings = (
  spark
    .table('instacart.user_ratings')
    .filter("split = 'calibration'")
    .select('user_id', 'product_id', 'normalized_purchases')
  )

# aggregate user ratings into per-user vectors
ratings_lists = (
  user_ratings
    .groupBy(user_ratings.user_id)
      .agg(
        collect_list(user_ratings.product_id).alias('index_list'),
        collect_list(user_ratings.normalized_purchases).alias('value_list')
        )
    )

# calculate vector size
vector_size = (
  spark
    .table('instacart.products')
    .groupBy()
      .agg( 
        (lit(1) + max('product_id')).alias('size')
        )
    )

# assemble ratings dataframe
ratings_vectors = (
  ratings_lists
    .crossJoin(vector_size)
    .withColumn(
      'ratings', 
      to_vector(
        vector_size.size, 
        ratings_lists.index_list, 
        ratings_lists.value_list
        )
      )
    .select(ratings_lists.user_id, 'ratings')
  )

display(ratings_vectors)

# COMMAND ----------

# MAGIC %md
# MAGIC これで、LSHテーブルを生成し、ユーザーをバケットに割り当てることができます。

# COMMAND ----------

# DBTITLE 1,LSHバケツを割り当てる(bucketLength=1)
bucket_length = 1
lsh_tables = 1

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# アルゴリズムをDatasetにfitさせる
fitted_lsh = lsh.fit(ratings_vectors)

# LSHバケツをユーザーにアサインする
hashed_vectors = (
  fitted_lsh
  .transform(ratings_vectors)  
).cache()

# このDataframeをSQLから参照できるようにtemp viewを作る
hashed_vectors.createOrReplaceTempView('hashed_vectors')

display(
  hashed_vectors 
)

# COMMAND ----------

# MAGIC %md 
# MAGIC LSHが何をしたのか、ハッシュ化された出力を見てみましょう。ハッシュフィールドには、各テーブルのバケットの割り当てを表すベクトルの配列が含まれています。 1つのハッシュテーブルでは、配列の中に1つのベクトルがあり、その値は0と-1の間で揺れ動きます。これを見やすくするために、そのベクトルから値を抽出してみましょう。
# MAGIC 
# MAGIC **注意** ここでは、ハッシュ化された値のベクトルからバケットIDを抽出するために、少しずさんな方法を取ります。 このようにすると、より簡潔になります。 また、今のところ*htable*フィールドは無視してください。

# COMMAND ----------

# DBTITLE 1,LSHバケツ割り当て結果を可視化する
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )  
# MAGIC SELECT *
# MAGIC FROM user_comparisons

# COMMAND ----------

# MAGIC %md 
# MAGIC ユーザーが2つのバケットに分かれているようです。 それぞれのバケツの中のユーザー数を見てみましょう。

# COMMAND ----------

# DBTITLE 1,バケツ毎のユーザー数
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*)
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ユーザーを2つのバケットに分けるだけで、必要なユーザー比較数を約200億人から約110億人に減らすことができました。

# COMMAND ----------

# DBTITLE 1,ユーザー比較数(計算回数)
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ハッシュテーブル内のバケットの数は、`bucketLength`引数によって制御されます。 `bucketLength`の値が低いほど、各バケットが対応するスペースが少なくなり、スペース内のすべてのユーザーを捕捉するために必要なバケットの数が多くなります。 このパラメータは、逆スロットルのようなものだと考えることができます。このパラメータを下げると、空間を切り分けるために使用される超平面の数が増え、結果として出力されるバケットの数が増えます。
# MAGIC 
# MAGIC [BucketedRandomProjectionLSH変換のコード](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/BucketedRandomProjectionLSH.scala)では、`bucketLength`を1～10の間に設定するのが最適であるとされていますが、`bucketLength`を1.0に設定すると、ユーザーを2つのバケットに分ける単一の超平面しか生成されないことがわかります。より多くのバケット数を生成するには、このパラメータ値を下げる必要があります。

# COMMAND ----------

# DBTITLE 1,LSHバケツを割り当てる (bucketLength=0.001)
bucket_length = 0.001
lsh_tables = 1

lsh = BucketedRandomProjectionLSH(
  inputCol = 'ratings', 
  outputCol = 'hash', 
  numHashTables = lsh_tables, 
  bucketLength = bucket_length
  )

# アルゴリズムをDatasetにfitさせる
fitted_lsh = lsh.fit(ratings_vectors)

# ユーザーにLSHバケツを割り当てる
hashed_vectors = (
  fitted_lsh
  .transform(ratings_vectors)  
).cache()

# このdataframeをSQLで利用できるようにtemp viewを作成する
hashed_vectors.createOrReplaceTempView('hashed_vectors')

# COMMAND ----------

# DBTITLE 1,バケツ毎のユーザ数
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*) as users
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket
# MAGIC ORDER BY htable, bucket

# COMMAND ----------

# DBTITLE 1,ユーザー比較の回数(計算回数)
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `BucketLength`の削減がデータ変換にどのような影響を与えたのか、まずはバケットごとのユーザー数を見てみましょう。 2つの大規模なバケットの代わりに、ユーザーが分類されるバケットが40個ほどになりました。これにより、必要なユーザーの比較数は約14億人に減り、これは良いことです。 しかし、これらのバケットに含まれるユーザーの分布を見てみましょう。
# MAGIC 
# MAGIC データサイエンティストは、美しいガウス曲線を好みますが、このシナリオでは違います。 ユーザーが中央のバケット（`bucket_id=-1`）に集中しているのは、「次元の呪い」を示しています。 簡単に言えば、多くのユーザーは、50,000の製品の特徴を持つ超空間の中で、互いによく似ており、空間の端に位置しているのです。そのため、バケットに含まれるユーザー数が偏ってしまい、比較対象となるユーザー数が多いバケットもあれば、比較対象となるユーザー数が少ないバケットもあるのです。 分散型のインフラを活用するために、比較回数を減らし、比較回数をより均等にすることが目的であれば、この問題を考慮する必要があります
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC しかし、`bucketLength`に話を戻すと、その値を下げるとバケット数が増えることがわかります。各バケットには、全体の超次元空間を分割するために生成された様々な超平面の間の空間に存在するユーザーが集められます。各超平面はランダムに生成されるので、バケット数を増やすことで、ランダムな超平面の数が増え、よく似た2人のユーザーが別々のバケットに分けられる可能性が高くなります。
# MAGIC 
# MAGIC この問題を解決するコツは、ユーザーをバケットに分割する作業を複数回行うことです。 それぞれの順列は、独立した別の *hash table* を生成するために使用されます。似たようなユーザーが別々のバケットに分割されるという問題は残りますが、2人の似たようなユーザーが、異なる（独立した）ハッシュテーブルで繰り返し別々のバケットに分割される確率は低くなります。 また、生成されたいずれかのハッシュテーブルにおいて、2人のユーザーが同じバケットに存在する場合、そのバケットは比較の対象となります。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_hash03x.png' width='600'>
# MAGIC 
# MAGIC もちろん、複数のハッシュテーブルを計算することは、本アプローチの計算負荷を増加させます。 また、すべてのハッシュテーブルで同じバケツに入っているユーザーが比較されることを考えると、実行しなければならないユーザー比較の数も増えます。適切なハッシュテーブルの数を決定するには、計算時間と類似したユーザーを「見逃す」ことに対する意欲のバランスを取ることが重要です。 この概念を探るために、バケットの長さを前回のコードブロックと同じにして、ハッシュテーブルの数を3に増やしてみましょう。

# COMMAND ----------

# DBTITLE 1,LSHバケツを割り当てる (bucketLength=0.001, numHashTables=3)
bucket_length = 0.001
lsh_tables = 3

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

# replace view with new hash data
hashed_vectors.createOrReplaceTempView('hashed_vectors')

display(
  hashed_vectors.select('user_id', 'hash')
  )

# COMMAND ----------

# DBTITLE 1,ハッシュテーブル/バケツ毎のユーザー数
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT 
# MAGIC   x.htable,
# MAGIC   x.bucket,
# MAGIC   COUNT(*) as users
# MAGIC FROM user_comparisons x
# MAGIC GROUP BY x.htable, x.bucket
# MAGIC ORDER BY htable, bucket

# COMMAND ----------

# DBTITLE 1,ユーザー比較の回数
# MAGIC %sql
# MAGIC 
# MAGIC WITH user_comparisons AS (
# MAGIC   SELECT
# MAGIC       a.user_id,
# MAGIC       b.htable,
# MAGIC       CAST(
# MAGIC         REGEXP_REPLACE(
# MAGIC           CAST(b.vector AS STRING),
# MAGIC           '[\\[\\]]',
# MAGIC           ''
# MAGIC           ) AS INT) as bucket
# MAGIC   FROM hashed_vectors a
# MAGIC   LATERAL VIEW posexplode(a.hash) b as htable, vector
# MAGIC   )
# MAGIC SELECT
# MAGIC   SUM(compared_to) as total_comparisons
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     x.user_id,
# MAGIC     COUNT(DISTINCT y.user_id) as compared_to
# MAGIC   FROM user_comparisons x
# MAGIC   INNER JOIN user_comparisons y
# MAGIC     ON x.user_id < y.user_id AND
# MAGIC        x.htable = y.htable AND
# MAGIC        x.bucket = y.bucket
# MAGIC   GROUP BY
# MAGIC     x.user_id
# MAGIC   )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ハッシュテーブルの数を増やすと、ユーザー比較を行うために調査する必要のあるバケットの数が増えます。バケット間のユーザー数の分布にわずかな違いが見られ、これらすべての影響を感じ取ることができます。合計すると、3つのテーブル間で必要なユーザー比較がかなり増加していることから、多くのお客様が3つのテーブル間で異なるバケットに分割されていることがわかります。
# MAGIC 
# MAGIC では、ハッシュテーブルの数は3つでいいのでしょうか？ 繰り返しになりますが、答えは簡単ではありません。処理時間と結果の正確さの間のトレードオフに関する簡潔な決定を行う必要があるからです。 ここでは、特定のユーザーを対象に、1人のユーザー（例えば`ser_id=148`）についてブルートフォース比較を徹底的に行い、その後、LSHを使って固定のバケット長0.001でテーブル数を変えた場合の結果を見てみましょう。

# COMMAND ----------

# MAGIC %md 
# MAGIC そして、ユーザー148のブルートフォース評価（他の200,000人のユーザーすべてに対して）です。

# COMMAND ----------

# DBTITLE 1,テストユーザーの網羅的な類似度の算出
# ratings for all users
ratings = (
  spark
    .sql('''
      SELECT
        user_id,
        COLLECT_LIST(product_id) as products_list,
        COLLECT_LIST(normalized_purchases) as ratings_list
      FROM (
        SELECT
          user_id,
          product_id,
          normalized_purchases
        FROM instacart.user_ratings
        WHERE split = 'calibration'
          )
      GROUP BY user_id
      ''')
  )

# assemble user A data
ratings_a = (            
  ratings
  .filter(ratings.user_id == 148) # limit to user 148
  .withColumnRenamed('user_id', 'user_a')
  .withColumnRenamed('products_list', 'indices_a')
  .withColumnRenamed('ratings_list', 'values_a')
)

# assemble user B data
ratings_b = (
  ratings
  .withColumnRenamed('user_id', 'user_b')
  .withColumnRenamed('products_list', 'indices_b')
  .withColumnRenamed('ratings_list', 'values_b')  
)

# calculate number of index positions required to hold product ids (add one to allow for index position 0)
size = spark.sql('''
  SELECT 1 + COUNT(DISTINCT product_id) as size 
  FROM instacart.products
''')

# cross join to associate every user A with user B
a_to_b = (
  ratings_a
  .crossJoin(ratings_b)   
).crossJoin(size)

# determine number of partitions per executor to keep partition count to 100,000 records or less
partitions_per_executor = 1 + int((a_to_b.count() / sc.defaultParallelism)/100000)

# redistribute the user-to-user data and calculate similarities
brute_force = (
  a_to_b
  .withColumn('id', monotonically_increasing_id())
  .withColumn('subset_id', expr('id % ({0} * {1})'.format(sc.defaultParallelism, partitions_per_executor)))
  .groupBy('subset_id')
  .applyInPandas(
    compare_users, 
    schema='''
      user_a int, 
      user_b int, 
      distance double
        ''').cache()    
)

display(
  brute_force
  .orderBy('distance', ascending=True)  
)

# COMMAND ----------

# MAGIC %md 
# MAGIC それでは、固定のバケット長と可変数のハッシュテーブルを使ってLSHルックアップを行ってみましょう。

# COMMAND ----------

# DBTITLE 1,ユーザー#148のベクタを抽出
# ユーザー#148のベクタを抽出

user_148_vector = (
  ratings_a
    .crossJoin(size)
    .withColumn('vector', to_vector('size','indices_a','values_a'))
  ).collect()[0]['vector']

user_148_vector

# COMMAND ----------

# DBTITLE 1,複数ハッシュテーブル化したLSHを用いて類似した隣人を特定する
bucket_length = 0.001

# initialize results with brute force results
results = brute_force

# initialize objects within loops
temp_lsh = []
temp_fitted_lsh = []
temp_hashed_vectors = []
temp_results = []

# loop through lsh table counts 1 through 10 ...
for i, lsh_tables in enumerate(range(1,11)):
  
  # generate lsh hashes
  temp_lsh += [
    BucketedRandomProjectionLSH(
    inputCol = 'ratings', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    )]

  temp_fitted_lsh += [temp_lsh[i].fit(ratings_vectors)]

  # calculate bucket assignments
  temp_hashed_vectors += [(
    temp_fitted_lsh[i]  
    .transform(ratings_vectors)    
  )]

  # lookup 100 neighbors for user 148
  temp_results += [(
    
    temp_fitted_lsh[i].approxNearestNeighbors(
      temp_hashed_vectors[i], 
      user_148_vector, 
      100, 
      distCol='distance_{0}'.format(str(lsh_tables).rjust(2,'0'))
    )
    .select('user_id', 'distance_{0}'.format( str(lsh_tables).rjust(2,'0')))
  )]
  
  # join results to prior results
  results = (
    results
    .join(
      temp_results[i], 
      on=results.user_b==temp_results[i].user_id, 
      how='outer'
    )
    .drop(temp_results[i].user_id)  
  )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 最後のセルのループの後半では、LSHルックアップをブルートフォースのデータセットに結合しました。 組み合わせた結果を見て、ハッシュテーブルを追加したときにユーザーの比較で何が起こっているかを確認してみましょう。

# COMMAND ----------

# DBTITLE 1,Compare Exhaustive Comparisons to LSH Comparisons at Different Table Counts
display(
  results
  .orderBy('distance', ascending=True)  
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC バケットの割り当てにはランダム性があるため、各回の結果は異なるかもしれませんが、より多くのテーブルが計算されるまで、より類似性の高いユーザーがLSHの結果に表示されないというケースが見られるはずです。この現象は、バケットの長さを長くして、より多くのユーザーを含むバケットを作ることで解決できます。

# COMMAND ----------

# DBTITLE 1,Identify Similar Neighbors using LSH with Differing Hash Table Counts and Higher Bucket Length
bucket_length = 0.01

# initialize results with brute force results
results = brute_force

# initialize objects within loops
temp_lsh = []
temp_fitted_lsh = []
temp_hashed_vectors = []
temp_results = []

# loop through lsh table counts 1 through 10 ...
for i, lsh_tables in enumerate(range(1,11)):
  
  # generate lsh hashes
  temp_lsh += [
    BucketedRandomProjectionLSH(
    inputCol = 'ratings', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    )]

  temp_fitted_lsh += [temp_lsh[i].fit(ratings_vectors)]

  # calculate bucket assignments
  temp_hashed_vectors += [(
    temp_fitted_lsh[i]   
    .transform(ratings_vectors)
  )]

  # lookup 100 neighbors for user 148
  temp_results += [(
    temp_fitted_lsh[i].approxNearestNeighbors(
      temp_hashed_vectors[i], 
      user_148_vector, 
      100, 
      distCol='distance_{0}'.format(str(lsh_tables).rjust(2,'0'))
      )
      .select('user_id', 'distance_{0}'.format( str(lsh_tables).rjust(2,'0')))
    )]
  
  # join results to prior results
  results = (
    results
      .join(
        temp_results[i], 
        on=results.user_b==temp_results[i].user_id, 
        how='outer'
        )
      .drop(temp_results[i].user_id)
      )
  
display(
  results  
  .orderBy('distance', ascending=True)  
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC バケットの長さを変更することで、少ないハッシュテーブルで類似のユーザーを見つけられる可能性が高くなることがわかります。 この2つの適切なバランスを見つけることは、精度がクエリのパフォーマンスとトレードオフの関係にあるため、科学というよりは芸術です。 また、これらの結果を比較するための網羅的なブルートフォース評価は、このノートの冒頭で説明したように実行可能ではありません。 このため、LSHのユーザーは[Malcom Slaney *et al.*'s in-depth exploration of these aspects of LSH tuning](https://www.slaney.org/malcolm/yahoo/Slaney2012%28OptimalLSH%29.pdf)を読み、これらの要素がどのように組み合わされて必要な結果をもたらすのか、直感的に理解することをお勧めします。

# COMMAND ----------

# DBTITLE 1,データセットのキャッシュを解放する
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
