# Databricks notebook source
# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC このノートブックの目的は、様々な製品メタデータフィールドに適用できる技術を検討し、推奨事項の基礎となる類似性を計算することです。カテゴリー階層データに焦点を当て、過去2回のノートブックで行った特徴抽出作業を継続して行います。この種のデータに基づく推薦は、単独で使用されることはほとんどありませんが、代わりに他の製品レコメンダーと組み合わせてスコアを調整し、製品階層に合わせて推薦を制限する必要があります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC このノートブックは **Databricks ML 7.3+ クラスタ** で実行する必要があります。

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, lit, count, countDistinct, col, array_join, expr, monotonically_increasing_id, explode

from typing import Iterator

from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH, HashingTF, Word2Vec
from pyspark.ml.clustering import LDA, KMeans, BisectingKMeans
from pyspark.ml.linalg import Vector, Vectors

from pyspark.sql import DataFrame

import mlflow.spark

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: Categoryデータの準備
# MAGIC 
# MAGIC 理想的なシナリオでは、製品階層が一貫した方法で整理され、すべてのエントリが共有のルートノードにロールアップされます。 このような構造はツリーと呼ばれ、ツリー構造内の製品の位置に基づいて製品の類似性を計算することができます。 [Thabet Slimani](https://arxiv.org/ftp/arxiv/papers/1310/1310.8059.pdf)は、ツリーで使用可能な様々な類似性メトリクスについて**優れた**レビューを提供しています。
# MAGIC 
# MAGIC 残念ながら、このデータセットに含まれる製品カテゴリーはツリーを形成していない。 それどころか、ある製品では子カテゴリーが親カテゴリーの下に位置していても、別の製品ではその位置が逆になっていたり、子カテゴリーが親を飛び越えて共通の祖先に直接転がり込んでいたりする矛盾が見られます。 このデータセットのカテゴリー階層の問題は簡単には修正できないため、このデータをツリーとして扱うことはできません。 とはいえ、類似性を評価するためにちょっとした工夫をすることはできます。
# MAGIC 
# MAGIC **注** もし、あなたの組織が管理しているデータセットがこれに当てはまるなら、マスターデータ管理ソリューションに投資することで、時間をかけてデータをより実用的な構造に移すことができるかもしれません。
# MAGIC 
# MAGIC カテゴリーデータを使い始めるために、まずいくつかの値を取得し、その構造を調べてみましょう。

# COMMAND ----------

# DBTITLE 1,Categoryデータの抽出
# retrieve descriptions for each product
categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category) > 0')
    .select('id', 'asin', 'category')
  )

num_of_categories = categories.count()
num_of_categories

# COMMAND ----------

display(
  categories
  )

# COMMAND ----------

# MAGIC %md カテゴリーデータは配列で構成されており，配列のインデックスは階層のレベルを示しています． 各レベルの名前に祖先の情報を付加することで、全体の構造の中で各レベルを一意に識別することができ、共通の名前を持つが異なる親や他の祖先の下に存在するカテゴリを互いに区別できるようになります。ここでは、pandas UDF を使用してこの作業を行います。
# MAGIC 
# MAGIC **注** レベル名が製品や機能の説明のようになっているレベルが多く見受けられます。 これが有効なデータなのか、ソースウェブサイトからのデータの誤認識なのかはわかりません。 このようなデータを避けるために、カテゴリーの階層を最大10レベルに制限し、100文字以上のレベル名がある場合は階層を解除することにします。 これらは任意の設定であり、あなたのデータセットに合わせて調整したり、破棄したりすることができます。

# COMMAND ----------

# DBTITLE 1,Rename Categories to Include Ancestry
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


categories = (
  spark
    .table('reviews.metadata')
    .filter('size(category)>0')
    .select('id','asin','category')
    .withColumn('category_clean', cleanse_category('category'))
  )

display(categories)

# COMMAND ----------

# MAGIC %md # Step 2: Category類似性を算出
# MAGIC 
# MAGIC レベル名が調整されたので、各製品のカテゴリーレベルをワンショットでエンコードしてみましょう。 この作業には、TF-IDF計算で使用したCountVectorizerを使用し、*binary*引数を*True*に設定して、すべての出力が0または1のいずれかになるようにします（いずれにしてもそうなるはずです）。 前述のように、この変換は、エントリーを、頻繁に発生する値の上位数に制限します。 製品カテゴリーに関する知識に基づいて、デフォルトから調整するかもしれませんが、今はこのままにしておきます。

# COMMAND ----------

# DBTITLE 1,CategoriesをOne-Hot Encodeする
categories_encoded = (
  HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )
    .transform(categories)
    .select('id','asin','category','category_onehot')
    .cache()
  )

display(
  categories_encoded
  )

# COMMAND ----------

# MAGIC %md  先ほどと同様に、レコードをバケットに分ける必要があります。 カテゴリの組み立て方から、2つのアイテムのトップレベルのカテゴリが同じでなければ、他の機能も重複しないことがわかっています。そこで、categories 配列の最初のメンバーに基づいて、製品をバケットにグループ化し、その結果を見てみましょう。

# COMMAND ----------

# DBTITLE 1,Group Items Based on Top Parent
roots = (
  categories_encoded
    .withColumn('root', expr('category[0]'))
    .groupBy('root')
      .agg(count('*').alias('members'))
    .withColumn('bucket', monotonically_increasing_id())
  )

categories_clustered = (
  categories_encoded
    .withColumn('root', expr('category[0]'))
    .join(roots, on='root', how='inner')
    .select('id','asin','category','category_onehot','bucket')
  )

display(roots.orderBy('members'))

# COMMAND ----------

# MAGIC %md この方法で問題に取り組むと、多少の歪んだ結果が得られますが、これはさらなる変換を行わなくても十分に対処可能です。
# MAGIC 
# MAGIC ここで、類似性に関する簡単な計算を行います。 類似性を距離（または角度）で測定した他の特徴とは異なり、これらの特徴は単純な[Jaccard類似性](https://en.wikipedia.org/wiki/Jaccard_index)スコアを使用して比較することができます。このスコアでは、重複するレベルの数を2つの製品間の異なるレベルの数で割ります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_jaccard_similarity2.png" width="250">
# MAGIC 
# MAGIC 前述のように、この作業はScalaの関数を使って行います。この関数をSpark SQLに登録し、Pythonで定義したDataFrameに対して利用できるようにします。

# COMMAND ----------

# DBTITLE 1,Jaccard Similarityを算出する関数を定義する
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.SparseVector
# MAGIC 
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC 
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)

# COMMAND ----------

# MAGIC %md サンプル製品に戻って、これがどのように機能するか見てみましょう。
# MAGIC 
# MAGIC **注意** ここでは、以前のノートブックで使用したのと同じサンプル製品を使用しています。

# COMMAND ----------

# DBTITLE 1,Retrieve Sample Product
sample_product = categories_clustered.filter("asin=='B01G6M7CLK'")
display(sample_product)

# COMMAND ----------

# DBTITLE 1,類似プロダクトを見つける
display(
  categories_clustered
    .withColumnRenamed('category_onehot', 'features_b')
    .join(sample_product.withColumnRenamed('category_onehot', 'features_a'), on='bucket', how='inner')
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .orderBy('similarity', ascending=False)
    .limit(100)
    .select(categories_clustered.id, categories_clustered.asin, categories_clustered.category, 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,キャッシュの削除
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md # Step 3: 類似性スコアを組み合わせてレコメンデーションを構築する
# MAGIC 
# MAGIC この時点では、タイトル、説明文、カテゴリの割り当てに基づいて、製品の類似性を比較する機能があります。 それぞれの機能を個別に使用してレコメンデーションを行うこともできますが、複数の要素から得られる情報を組み合わせることで、より目的に沿った結果を得ることができるかもしれません。 例えば、タイトルに基づいた推薦は文字数が多く、説明文に基づいた推薦は漠然としていますが、これらを組み合わせることで、説得力のある推薦を行うための適切なバランスが得られるかもしれません。また、これらの機能セットの一方または両方を使用して類似性を計算することで、ここに示すように、カテゴリの類似性メトリックと組み合わせることができ、カテゴリ階層の同じ部分または関連する部分から製品を推薦することができます。
# MAGIC 
# MAGIC これらのシナリオでは、類似性測定基準をどのように組み合わせるのがベストかを検討する必要があります。一つの単純な方法は、それぞれの類似性スコアを単純に掛け合わせて、単純な複合スコアを形成することです。また、それぞれの類似性スコアに重み付けをして、その重み付けされた値を互いに加算するという方法もあります。複数のレコメンダーが生成した類似性スコアを使って統一的なレコメンデーションを行う方法は、工夫次第でもっと多くのアプローチが考えられます。 これらの入力をどのように組み合わせるかを考えるポイントは、1つの方法を指定することではなく、目的に応じて、いくつかのアプローチが妥当であることを強調することです。
