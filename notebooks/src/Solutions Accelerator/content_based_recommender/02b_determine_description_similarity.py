# Databricks notebook source
# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC このノートブックの目的は、製品間の類似性を計算するために、製品の説明からどのように特徴を抽出するかを検討することです。(他のメタデータフィールドについては、このノートブックに付随するノートブックで検討します)。これらの類似性は、**関連製品**を推奨するための基礎として使用されます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC このノートブックは **Databricks ML 7.3+ クラスタ** で実行する必要があります。

# COMMAND ----------

# MAGIC %md **注意** このノートブックを実行するクラスターは、NLTK WordNetコーパスとAveraged Perceptron Taggerをインストールする[cluster-scoped initialization script](https://docs.databricks.com/clusters/init-scripts.html?_ga=2.158476346.1681231596.1602511918-995336416.1592410145#cluster-scoped-init-script-locations)を使用して作成する必要があります。 次のセルを使用してこのようなスクリプトを生成することができますが、スクリプトに依存するコードを実行する前に、このスクリプトをクラスターに関連付ける必要があります。

# COMMAND ----------

# DBTITLE 1,Cluster Init Scriptの生成
dbutils.fs.mkdirs('dbfs:/databricks/scripts/')

dbutils.fs.put(
  '/databricks/scripts/install-nltk-downloads.sh',
  '''#!/bin/bash\n/databricks/python/bin/python -m nltk.downloader wordnet\n/databricks/python/bin/python -m nltk.downloader averaged_perceptron_tagger''', 
  True
  )

# alternatively, you could install all NLTK elements with: python -m nltk.downloader all

# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/install-nltk-downloads.sh')
  )

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

import nltk

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: Descriptionデータの準備
# MAGIC 
# MAGIC タイトルは便利ですが、情報量は限られています。 長い説明文（データセットの*description*フィールドにあるような）は、似たようなアイテムを識別するために使用できる詳細な情報を提供します。
# MAGIC 
# MAGIC しかし、比較対象となる単語が多いということは、処理するデータが増え、複雑になることを意味します。 そこで、テキストをより狭い範囲の「トピック」や「コンセプト」に凝縮して、比較の基礎とする次元削減技術を利用することができます。
# MAGIC 
# MAGIC この方向性を探るために、説明データを格納する配列をフラット化し、先ほどのようにテキスト内の単語をトークン化してみましょう。

# COMMAND ----------

# DBTITLE 1,Descriptionsの抽出
# retrieve descriptions for each product
descriptions = (
  spark
    .table('reviews.metadata')
    .filter('size(description) > 0')
    .withColumn('descript', array_join('description',' '))
    .selectExpr('id', 'asin', 'descript as description')
  )

num_of_descriptions = descriptions.count()
num_of_descriptions

# COMMAND ----------

# present descriptions
display(
  descriptions
  )

# COMMAND ----------

# DBTITLE 1,Descriptionsから単語を抽出
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='description', 
    outputCol='words'
    )

description_words = tokenizer.transform(descriptions)

display(description_words)

# COMMAND ----------

# MAGIC %md # Step 2a: LDAトピック特徴量を抽出
# MAGIC 
# MAGIC それでは、[Latent Dirichlet Allocation (LDA)](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)を使って、記述をトピックの凝縮されたセットに還元する方法を探ってみましょう。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_LDA.gif" width="500">
# MAGIC 
# MAGIC LDAの背後にある数学は複雑ですが、技術自体は非常に理解しやすいものです。 簡単に言うと、説明文に含まれる単語の共起を調べるのです。 ある程度の規則性を持って互いに出現する単語の「クラスター」は、「トピック」を表していますが、人間にはそのようなトピックを理解するのは少し難しいかもしれません。しかし、各説明文を、説明文全体から発見された各トピックとの整合性に基づいてスコアリングすることができます。 これらのトピックごとのスコアは、データセット内の類似したドキュメントを見つけるための基礎となります。
# MAGIC 
# MAGIC LDAの計算を行うためには、タイトルデータと同じように、レマタイズを用いて説明文中の単語を標準化する必要があります。

# COMMAND ----------

# DBTITLE 1,単語の標準化
# declare the udf
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

# use function to convert words to lemmata
description_lemmata = (
  description_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'description', 'lemmata')
    ).cache()

display(description_lemmata)

# COMMAND ----------

# MAGIC %md 次に、データセット内の単語の出現数を数えなければなりません。 タイトルに含まれるよりも多くの単語を扱っているので、この作業には[HashingTF変換](https://spark.apache.org/docs/2.2.0/ml-features.html#tf-idf)を使います。 
# MAGIC 
# MAGIC HashingTFとCountVectorizerは同じ作業を行っているように見えますが、HashingTF変換では処理を高速化するためのショートカットを使用しています。説明文中のすべての単語から個別の単語リストを作成し、それぞれについて用語頻度スコアを計算し、結果として得られるベクトルをこれらの単語のうち上位に出現するものに限定する代わりに、HashingTF変換は各単語を整数値にハッシュ化し、そのハッシュ化された値をベクトル内の単語のインデックスとして使用します。これにより、単語のルックアップテーブルを作成し、それに対してカウントを実行するというステップを省略することができます。 代わりに、ハッシュを計算して、関連するインデックスの位置の値に1を加えるだけでよいのです。 その代償として、ハッシュの衝突が起こり、2つの無関係な単語があたかも同じものであるかのようにカウントされる事態が発生します。 パフォーマンスを大幅に向上させるために、多少のずさんさの可能性を受け入れるのであれば、HashingTF変換がお勧めです。

# COMMAND ----------

# DBTITLE 1,単語のカウント
# get word counts from descriptions
description_tf = (
  HashingTF(
    inputCol='lemmata', 
    outputCol='tf',
    numFeatures=262144  # top n words to consider
    )
    .transform(description_lemmata)
  )

display(description_tf)

# COMMAND ----------

# MAGIC %md 単語数が確定したので、今度はLDAを適用してトピックを定義し、そのトピックに対する記述をスコアリングします。 反復回数やクラスタの大きさによっては、このステップの完了に時間がかかる場合があります。 この点を考慮して、データセット全体の25%のランダムサンプルを使用してLDAトピックを*学習*していることに注意してください。これにより、計算時間を短縮しながら有効なトピックを得ることができます。

# COMMAND ----------

# DBTITLE 1,LDAを適用
# identify LDA topics & score descriptions against these
description_lda = (
  LDA(
    k=100, 
    maxIter=20,
    featuresCol='tf',
    optimizer='online' # use the online optimizer for scalability
    )
    .fit(description_tf.sample(False, 0.25)) # train on a random sample of the data
    .transform(description_tf) # transform all the data
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_lda', ignore_errors=True)

# persist as delta table
(
  description_lda
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_lda')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_lda`').select('id','asin','description','topicDistribution')
  )

# COMMAND ----------

# MAGIC %md LDAスコアは、記述の類似性を評価するための特徴となります。 以前のように、このステップに進む前に、これらを正規化する必要があります。

# COMMAND ----------

# DBTITLE 1,LDA特徴量の正規化
description_lda = spark.table('DELTA.`/mnt/reviews/tmp/description_lda`')

description_lda_norm = (
  Normalizer(inputCol='topicDistribution', outputCol='features', p=2.0)
    .transform(description_lda)
  ).cache()

display(description_lda_norm.select('id','asin','description', 'features'))

# COMMAND ----------

# MAGIC %md 正規化された特徴量を使って類似性を計算する方法を見る前に、別の次元削減技術を見てみましょう。

# COMMAND ----------

# MAGIC %md # Step 2b: Extract Word2Vec *Concept* Features
# MAGIC 
# MAGIC LDAは、説明文の中の任意の場所にある単語を使って、発見されたトピックとの関係をスコアリングします。つまり、説明文中の単語の順序は考慮されません。 一方、[Word2Vec](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/)では、記述の中の*概念*を得るために、単語の近接性を調べます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_w2v.gif" width="600">
# MAGIC 
# MAGIC **注意** Word2Vecはトークン化以外の前処理を必要としません。 また、Word2Vecの実行には時間がかかるため、データの25%のみでモデルをフィッティングしていることに注意してください。

# COMMAND ----------

# DBTITLE 1,Word2Vecの適用
description_words = description_words.cache()

# generate w2v set
description_w2v =(
  Word2Vec(
    vectorSize=100,
    minCount=3,              # min num of word occurances required for consideration
    maxSentenceLength=1000,  # max num of words in description to consider
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
    .fit(description_words.sample(False, 0.25))
    .transform(description_words)
  )

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_w2v', ignore_errors=True)

# persist as delta table
(
  description_w2v
   .write
   .format('delta')
   .mode('overwrite')
   .save('/mnt/reviews/tmp/description_w2v')
  )

display(
  spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`').select('id','asin','description','w2v')
  )

# COMMAND ----------

# MAGIC %md As with our LDA-derived features, our Word2Vec features require normalization:

# COMMAND ----------

# DBTITLE 1,Word2Vecの正規化
description_w2v = spark.table('DELTA.`/mnt/reviews/tmp/description_w2v`')

description_w2v_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(description_w2v)
  ).cache()

display(description_w2v_norm.select('id','asin','description', 'w2v', 'features'))

# COMMAND ----------

# MAGIC %md # Step 3: Descriptionの類似性の算出
# MAGIC 
# MAGIC では、LDAやWord2Vecの機能を使って、似たようなアイテムを見つけるにはどうすればいいのでしょうか？ 前回はLSHを使いましたが、今回も同じように使えます。 しかし、もう一つの手法として、k-meansクラスタリングを使うこともできます。
# MAGIC 
# MAGIC k-means クラスタリングでは、特徴の類似性に基づいて商品をクラスタに割り当てます。 クラスタを割り当てることで、類似した製品の検索をクラスタ内の製品に限定することができます（LSHを使って共有バケット内の製品に類似性検索を限定するのと同じです）。 Sparkでは、従来のk-meansとbisecting k-meansという2つの基本的なオプションがあります。どちらを使ってもいいのですが、結果は異なります。 どちらを選んでも、従来の[elbow technique](https://bl.ocks.org/rpgove/0060ff3b656618e9136b)を使って最適なクラスタ数を特定することができますが、結果セットに対するクエリのパフォーマンスを考慮することも重要です。 ここでは、50クラスタを選択します。これは、私たちのデータを合理的に分割することができると思われるからです。 クラスタリングをこのような言葉で表現することはあまりありませんが、このシナリオでのクラスタリングの適用は、LSHのような近似的な手法であると考えることが重要です。
# MAGIC 
# MAGIC **NOTE** クラスタ化/バケット化されたデータは、後のノートブックで再利用するためにDelta Lakeに永続化されています。さらに、同様の再利用のためにmlflowを使ってクラスタリングモデルを永続化しています。

# COMMAND ----------

# DBTITLE 1,Assign Descriptions to Clusters
clustering_model = (
  BisectingKMeans(
    k=50,
    featuresCol='features', 
    predictionCol='bucket',
    maxIter=100,
    minDivisibleClusterSize=100000
    )
    .fit(description_w2v_norm.sample(False, 0.25))
  )

descriptions_clustered = (
  clustering_model
    .transform(description_w2v_norm)
  )

# persist the clustering model for next notebook
with mlflow.start_run():
  mlflow.spark.log_model(
    clustering_model, 
    'model',
    registered_model_name='description_clust'
    )
  
# persist this data for the next notebook
shutil.rmtree('/dbfs/mnt/reviews/tmp/description_sim', ignore_errors=True)
(
  descriptions_clustered
    .write
    .format('delta')
    .mode('overwrite')
    .save('/mnt/reviews/tmp/description_sim')
  )

display(
  descriptions_clustered
    .groupBy('bucket')
    .agg(count('*').alias('descriptions'))
    .orderBy('bucket')
  )

# COMMAND ----------

# MAGIC %md データがクラスターに割り当てられていれば、類似した製品を見つけることは非常に簡単です。 必要なのは、各クラスタ/バケット内での徹底的な比較です。
# MAGIC 
# MAGIC LSHを使用した場合、ベクトル間のユークリッド距離の計算など、この作業は自動的に行われます。 ここでは、カスタム関数を使用して距離計算を行う必要があります。
# MAGIC 
# MAGIC 2つのベクトル間のユークリッド距離計算は、かなり簡単に実行できます。 しかし、以前使用したpandasのUDFには、ネイティブフォーマットのベクトルを受け入れる手段がありません。 しかし，Scalaにはあります． 2つのベクトル間のユークリッド距離を計算するScala関数をSpark SQLエンジンに登録することで、後のセルで紹介するように、Pythonを使ってSpark DataFrameのデータに関数を簡単に適用することができます。

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

# MAGIC %md それでは、サンプル製品と同じバケツ（クラスタ）に入っているアイテムに限定して、レコメンデーションを行ってみましょう。
# MAGIC 
# MAGIC **NOTE** 前回のノートで使用したサンプル製品と同じものを使用します。

# COMMAND ----------

# DBTITLE 1,サンプル製品の取得
sample_product = descriptions_clustered.filter('asin==\'B01G6M7CLK\'')

display(sample_product)

# COMMAND ----------

# DBTITLE 1,似たような記述を検索する
display(
  descriptions_clustered
    .withColumnRenamed('features', 'features_b')
    .join(sample_product.withColumnRenamed('features', 'features_a'), on='bucket', how='inner')  # join on bucket/cluster
    .withColumn('distance', expr('euclidean_distance(features_a, features_b)')) # calculate distance
    .withColumn('raw_sim', expr('1/(1+distance)')) # convert distance to similarity score
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    .orderBy('distance', ascending=True)
    .limit(100) # get top 100 recommendations
    .select(descriptions_clustered.id, descriptions_clustered.asin, descriptions_clustered.description, 'distance', 'similarity')
    )

# COMMAND ----------

# DBTITLE 1,キャッシュの削除
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md TF-IDFでスコアリングしたタイトルとは異なり、説明文のマッチングの根拠は、データを単純に見ただけでは少しわかりにくいものです。 トピック」や「コンセプト」という概念は、単純なワードカウントによるスコアとは異なり、少し捉えにくいものです。 しかし、説明文を見てみると、なぜある説明文が他の説明文よりも似ていると考えられるのかがわかります。さらに、LDAとWord2Vecの特徴量を生成する際には、製品の重要な側面に直接結びつく可能性が高い文章の部分であるため、説明文の最初からより少ない数の単語に制限することを検討することができます。 Word2Vecでは、簡単な引数の設定でこれを行います。 LDAの場合は、レマタイズを行う前に、トークン化された単語を切り詰めるステップを追加する必要があります。
