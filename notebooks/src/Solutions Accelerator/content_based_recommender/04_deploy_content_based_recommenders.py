# Databricks notebook source
# MAGIC %md このノートブックの目的は、以前のノートブックで開発されたコンテンツベースのレコメンダーがどのように運用されるかを探ることです。このノートブックは **Databricks ML 7.3+ クラスタ** で実行する必要があります。

# COMMAND ----------

# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC データ量、データ変更の頻度、レコメンデーションに期待されるサービスレベル、レコメンデーションが推進すべきビジネス目標などを慎重に検討した上で、レコメンデーションソリューションを構築する必要があります。とはいえ、多くのレコメンダーが使用している設計上の選択肢を想定し、これらを検討するための出発点となるデモンストレーション用の展開アーキテクチャを構築することも可能です。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_feature_pipelines5.png" width="700">
# MAGIC 
# MAGIC このアーキテクチャを実現するためには、これまでのノートで紹介してきたフィーチャーエンジニアリングのロジックを、パイプラインと呼ばれる管理可能な単位にまとめる必要があります。 パイプラインには生の情報が流入し、変換された機能が流出します。 ここでは、前回のノートで示したように、タイトル、説明、カテゴリーに関連する機能セットのそれぞれについて、3つのパイプラインを実装します。それぞれのパイプラインから出力されたフィーチャーは、別々のテーブルに格納されます。このようなパターンを採用することで、パイプラインに対する変更を別々にスケジュールしたり、展開したりすることが容易になります。同様に、ユーザープロファイル用のパイプラインも構築する必要があります。 ここで重要なのは、このパイプラインが、ユーザーからのフィードバック（評価という形で）と、他のパイプラインで得られた機能の両方を利用するということです。
# MAGIC 
# MAGIC フィーチャーの生成は、レコメンデーションソリューションが取り組むべき最初の課題です。次の課題は、レコメンデーション自体の生成です。 1つのアプローチは、製品の一部または全部について、「関連製品」のレコメンデーションをあらかじめ計算しておくことです。また、これらのレコメンデーションを動的に生成し、キャッシュして再利用する方法もあります。ここでは、事前計算のパターンに注目します。

# COMMAND ----------

# MAGIC %md **NOTE** このノートブックを実行するためのクラスターは、NLTK WordNetコーパスとAveraged Perceptron Taggerをインストールする[cluster-scoped initialization script](https://docs.databricks.com/clusters/init-scripts.html?_ga=2.158476346.1681231596.1602511918-995336416.1592410145#cluster-scoped-init-script-locations)を使用して作成する必要があります。 以下のセルを使用してこのようなスクリプトを生成することができますが、スクリプトに依存するコードを実行する前に、このスクリプトをクラスターに関連付ける必要があります。

# COMMAND ----------

# DBTITLE 1,Cluster Init Scriptの生成
dbutils.fs.mkdirs('dbfs:/databricks/scripts/')

dbutils.fs.put(
  '/databricks/scripts/install-nltk-downloads.sh',
  '''
  #!/bin/bash\n/databricks/python/bin/python -m nltk.downloader wordnet\n/databricks/python/bin/python -m nltk.downloader averaged_perceptron_tagger''', 
  True
  )

# alternatively, you could install all NLTK elements with: python -m nltk.downloader all

# show script content
print(
  dbutils.fs.head('dbfs:/databricks/scripts/install-nltk-downloads.sh')
  )

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
from pyspark import keyword_only
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.ml import Transformer, Pipeline
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH, SQLTransformer, HashingTF, Word2Vec
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.stat import Summarizer

import nltk

import mlflow.spark

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf, expr, col, lit, struct, collect_list, count
from typing import Iterator
from pyspark.sql.types import *

from delta.tables import *

import pandas as pd

import shutil
import os

# COMMAND ----------

# MAGIC %md # Step 1: 製品機能パイプラインの構築
# MAGIC 
# MAGIC 当社のコンテンツ・レコメンダーは、製品メタデータの*title*、*description*、*category*フィールドから得られる3つの特徴セットを使用しています。これらの特徴セットはそれぞれ、特定の順序で適用されなければならない一連の様々な変換によって作成されます。これらの変換に関する情報を保持するために、それらをパイプラインに整理し、サイクル間で再利用できるようにフィットさせて保存することができます。最初に構築するのは、タイトル変換のパイプラインです。
# MAGIC 
# MAGIC タイトル データに対して実行する基本的な変換手順は次のとおりです。
# MAGIC 1. タイトルをトークン化して単語のバッグを生成する。
# MAGIC 2. 単語の袋をレンマタイズする
# MAGIC 3. トークンの頻度を計算する
# MAGIC 4. TF-IDFスコアの算出
# MAGIC 5. TF-IDFスコアの正規化
# MAGIC 
# MAGIC これらのステップのうち、1つを除いたすべてのステップは、すぐに使えるトランスフォーマーで対応できます。 レマタイズのステップでは、カスタムトランスフォーマーを書く必要があります。 その手順は、[this post](https://stackoverflow.com/questions/32331848/create-a-custom-transformer-in-pyspark-ml)にうまく紹介されており、次のセルで実装されています。 カスタムトランスフォーマークラスの*_transform*メソッドは、以前のノートブックで作成したUDFをコピー＆ペーストしたものであることに注意してください。

# COMMAND ----------

# DBTITLE 1,Lemmatizationのためのカスタムトランスフォームの定義
class NLTKWordNetLemmatizer(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(NLTKWordNetLemmatizer, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, lemmatize_words(in_col))

# COMMAND ----------

# MAGIC %md カスタムトランスフォーマーが完成したので、次はタイトルパイプラインを構築してみましょう。

# COMMAND ----------

# DBTITLE 1,タイトルパイプラインの組み立て
# step 1: tokenize the title
title_tokenizer = RegexTokenizer(
  minTokenLength=2, 
  pattern='[^\w\d]', 
  inputCol='title', 
  outputCol='words'
  )

# step 2: lemmatize the word tokens
title_lemmatizer = NLTKWordNetLemmatizer(
  inputCol='words',
  outputCol='lemmata'
  )

# step 3: calculate term-frequencies
title_tf = CountVectorizer(
  inputCol='lemmata', 
  outputCol='tf',
  vocabSize=262144
  )
  
# step 4: calculate inverse document frequencies
title_tfidf = IDF(
  inputCol='tf', 
  outputCol='tfidf'
  )
  
# step 5: normalize tf-idf scores
title_normalizer = Normalizer(
  inputCol='tfidf', 
  outputCol='tfidf_norm', 
  p=2.0
  )

# step 6: assign titles to buckets
title_lsh = BucketedRandomProjectionLSH(
  inputCol = 'tfidf_norm', 
  outputCol = 'hash', 
  numHashTables = 5, 
  bucketLength = 0.0001
  )

# assemble pipeline
title_pipeline = Pipeline(stages=[
  title_tokenizer,
  title_lemmatizer,
  title_tf,
  title_tfidf,
  title_normalizer,
  title_lsh
  ])

# COMMAND ----------

# MAGIC %md これで、パイプラインを適合させ、変換を実行して、期待通りに動作することを確認できます。 パイプラインの適合は、後のセルでパイプラインを永続化する前に実際に必要な唯一のステップであることに注意してください。

# COMMAND ----------

# DBTITLE 1,Fit & Test Titles Pipeline
# retrieve titles
titles = (
  spark
    .table('reviews.metadata')
    .repartition(sc.defaultParallelism)
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

# fit pipeline to titles
title_fitted_pipeline = title_pipeline.fit(titles)

# present transformed data for validation
display(
  title_fitted_pipeline.transform(titles.limit(1000))
  )

# COMMAND ----------

# MAGIC %md 次に、パイプラインの永続化に目を向けてみましょう。 フィットしたパイプラインを永続化するために、[mlflow registry](https://mlflow.org/docs/1.4.0/model-registry.html)を利用します。
# MAGIC 
# MAGIC **注意** mlflow registryは、モデルがテストされ、時間の経過とともにステージ間を移動するMLOpsのワークフローを実現するために設計されています。 ロギング後すぐにパイプラインを取得しようとすると、エラーが発生する場合があります。永続化と取得のトラブルシューティングを行っている場合は、パイプラインオブジェクトが適切に利用できるように、ステップ間でレジストリに数秒の時間を与えるようにしてください。

# COMMAND ----------

# DBTITLE 1,フィットしたタイトルのパイプラインを持続させる
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    title_fitted_pipeline, 
    'pipeline',
    registered_model_name='title_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md ここで、記述データのパイプラインを作成します。 ここでは、以前のノートブックで検討した Word2Vec の機能を使用します。 パイプラインは、以下のステップで構成されます:</p>
# MAGIC 
# MAGIC 1. 説明文を配列から文字列に変換
# MAGIC 2. 説明をトークン化する
# MAGIC 2. Word2Vec変換の適用
# MAGIC 3. Word2Vecスコアの正規化
# MAGIC 4. KMeansクラスターの計算
# MAGIC 
# MAGIC このパイプラインでは、カスタムトランスフォーマーは必要ありませんが、以前にクエリで対処した最初のステップを処理するために、[SQLトランスフォーマー](https://spark.apache.org/docs/latest/ml-features#sqltransformer)を導入する必要があります。
# MAGIC 
# MAGIC **注** Word2Vecで考慮する単語は、事前の関連ノートブックで議論された情報ごとに、最初の200語に制限しています。また、SQLトランスフォーマーはデータセットのすべての入力フィールドにアクセスできますが、SELECTリストで指定されたものだけを渡すことを指摘しておきます。

# COMMAND ----------

# DBTITLE 1,Descriptionのパイプラインの構築
# step 1: flatten the description to form a single string
descript_flattener = SQLTransformer(
    statement = '''
    SELECT id, array_join(description, ' ') as description 
    FROM __THIS__ 
    WHERE size(description)>0'''
    )

# step 2: split description into tokens
descript_tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]', 
    inputCol='description', 
    outputCol='words'
    )

# step 3: convert tokens into concepts
descript_w2v =  Word2Vec(
    vectorSize=100,
    minCount=3,              
    maxSentenceLength=200,  
    numPartitions=sc.defaultParallelism*10,
    maxIter=20,
    inputCol='words',
    outputCol='w2v'
    )
  
# step 4: normalize concept scores
descript_normalizer = Normalizer(
    inputCol='w2v', 
    outputCol='features', 
    p=2.0
    )

# step 5: assign titles to buckets
descript_cluster = BisectingKMeans(
    k=50,
    featuresCol='features', 
    predictionCol='bucket',
    maxIter=100,
    minDivisibleClusterSize=100000
    )

# assemble the pipeline
descript_pipeline = Pipeline(stages=[
    descript_flattener,
    descript_tokenizer,
    descript_w2v,
    descript_normalizer,
    descript_cluster
    ])

# COMMAND ----------

# DBTITLE 1,モデル学習・テスト
# retrieve descriptions
descriptions = (
    spark
      .table('reviews.metadata')
      .select('id','description')
      .repartition(sc.defaultParallelism)
      .sample(False, 0.25)
    )

# fit pipeline
descript_fitted_pipeline = descript_pipeline.fit(descriptions)

# verify transformation
display(
  descript_fitted_pipeline.transform(descriptions.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,学習済みDescriptionパイプラインの永続化
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    descript_fitted_pipeline, 
    'pipeline',
    registered_model_name='descript_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md そして、今度はカテゴリーのパイプラインに取り組みます。 このパイプラインでは、以下のステップを実行します。
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC 1. リネージを含むようにカテゴリー レベルを変更する
# MAGIC 2. カテゴリ レベルのワンショット エンコーディングを実行する
# MAGIC 3. ルート・メンバーを特定する
# MAGIC 
# MAGIC これらのステップの最初の部分は、カスタム・トランスフォーマーによって処理されます。これらのステップの最後は、SQL トランスフォーマーによって処理されます。

# COMMAND ----------

# DBTITLE 1,カスタムトランスフォーマーの定義
# define custom transformer to flatten the category names to include lineage
class CategoryFlattener(
  Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable
  ):
 
  @keyword_only
  def __init__(self, inputCol=None, outputCol=None):
    super(CategoryFlattener, self).__init__()
    kwargs = self._input_kwargs
    self.setParams(**kwargs)
  
  @keyword_only
  def setParams(self, inputCol=None, outputCol=None):
    kwargs = self._input_kwargs
    return self._set(**kwargs)

  def setInputCol(self, value):
    return self._set(inputCol=value)
    
  def setOutputCol(self, value):
    return self._set(outputCol=value)
  
  def _transform(self, dataset):
    
    # copy-paste of previously defined UDF
    # =========================================
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
    # =========================================
    
    t = ArrayType(StringType())
    out_col = self.getOutputCol()
    in_col = dataset[self.getInputCol()]
    
    return dataset.withColumn(out_col, cleanse_category(in_col))

# COMMAND ----------

# DBTITLE 1,Categorieパイプラインの構築
# step 1: flatten hierarchy
category_flattener = CategoryFlattener(
    inputCol='category',
    outputCol='category_clean'
    )

# step 2: one-hot encode hierarchy values
category_onehot = HashingTF(
    inputCol='category_clean', 
    outputCol='category_onehot',
    binary=True
    )

# step 3: assign bucket
category_cluster = SQLTransformer(
    statement='SELECT id, category[0] as bucket, category, category_clean, category_onehot FROM __THIS__ WHERE size(category)>0'
    )

# assemble pipeline
category_pipeline = Pipeline(stages=[
    category_flattener,
    category_onehot,
    category_cluster
    ])

# COMMAND ----------

# DBTITLE 1,Categoriesパイプラインのモデル学習・テスト
# retrieve categories
categories = (
    spark
      .table('reviews.metadata')
      .select('id','category')
      .repartition(sc.defaultParallelism)
    )

# fit pipeline
category_fitted_pipeline = category_pipeline.fit(categories)

# test the transformation
display(
  category_fitted_pipeline.transform(categories.limit(1000))
  )

# COMMAND ----------

# DBTITLE 1,学習済みCategorieパイプラインの永続化
# persist pipeline
with mlflow.start_run():
  mlflow.spark.log_model(
    category_fitted_pipeline, 
    'pipeline',
    registered_model_name='category_fitted_pipeline'
    )

# COMMAND ----------

# MAGIC %md # Step 2: 製品特徴量の生成
# MAGIC 
# MAGIC パイプラインが定義され、トレーニングされ、永続化されたことで、今度はレコメンデーションを構築するためのフィーチャーの生成に集中できるようになりました。 新しいパイプラインを導入する際には、新しい情報を使って機能テーブルを初期化する必要があります。 この初期化を行った後は、メタデータテーブルに新製品が追加されたり、製品のメタデータが変更されたりした場合に、パイプラインを再実行する必要があります。
# MAGIC 
# MAGIC 理想的には、製品のメタデータテーブルに、これらの変更を確実に検出できる「変更された日付と時刻」の値があればいいのですが、そのようなフィールドはありません。 しかし、私たちのメタデータ・テーブルにはそのようなフィールドがないため、このロジックを実装することはできません。 とはいえ、変更検出を行う能力を持つ人のために、各機能セットのテーブルをどのように初期化し、更新するかを紹介します。

# COMMAND ----------

# MAGIC %md パイプラインを実行する最初のステップは、mlflow からパイプライン・オブジェクトを取得することです。 先ほど、パイプラインオブジェクトを mlflow のレジストリに保存しました。 このレジストリを利用することで、パイプラインオブジェクトを*Staging*、*Production*、*Archived*の各ステージに移動させることができ、このプロセスのためにパイプラインオブジェクトの現在の*Production*バージョンを取得することが簡単になります。 つまり、パイプラインをプログラム的に*Production*ステージに移動させておらず（これは通常、正式な評価を伴う別のプロセスで実行されるため）、そのために、現在*None*ステージに存在する各パイプラインの最後のバージョンを取得しています。(繰り返しになりますが、これは本番シナリオでは通常行われない方法です。)
# MAGIC 
# MAGIC ここでは、タイトルのパイプラインを例に説明します。
# MAGIC 
# MAGIC **注** この次のセルはエラーを生成します。

# COMMAND ----------

# DBTITLE 1,Titleパイプラインの検索を試す
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'  # models:/<model name defined at persistence>/<stage>
    )

# COMMAND ----------

# MAGIC %md 大きな冗長警告が生成されますが、これは無視して構いません。 ここで注目したいのは、カスタムトランスフォーマーのクラスが見つからないという**AttributeError**です。つまり、*module '__main__' has no attribute 'NLTKWordNetLemmatizer'*ということです。 パイプラインがクラス定義を見つけられないことは、[*leaky pipeline* problem](https://rebeccabilbro.github.io/module-main-has-no-attribute/)と呼ばれることがあり、Spark MLがエミュレートしている*sklearn*でよく見られる問題です。
# MAGIC 
# MAGIC この問題を解決するには2つの方法があります。</p>
# MAGIC 
# MAGIC 1. パイプラインオブジェクトを取得するすべてのコードにクラス定義を含め、トップレベル環境に認識させる。
# MAGIC 2. クラスタに添付できる *whl* ファイルを作成する。
# MAGIC 
# MAGIC whl*ファイルの作成については[こちら](https://packaging.python.org/guides/distributing-packages-using-setuptools/)（および[こちら](https://medium.com/swlh/beginners-guide-to-create-python-wheel-7d45f8350a94)）、そのファイルをDatabricksクラスタにアタッチするためのテクニックについては[こちら](https://docs.databricks.com/libraries/index.html)を参照してください。これは、よりエレガントな解決策ですが、ノートブックの範囲内では簡単に実演できない能力を必要とします。そのため、ここでは不完全な最初の解決策で問題を解決します。

# COMMAND ----------

# DBTITLE 1,カスタムのTransformerを見るためにnotebook上の環境を固定する
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'NLTKWordNetLemmatizer', NLTKWordNetLemmatizer)

# COMMAND ----------

# DBTITLE 1,Titleパイプラインの検索
# retrieve pipeline from registry
retrieved_title_pipeline = mlflow.spark.load_model(
    model_uri='models:/title_fitted_pipeline/None'
    )

# COMMAND ----------

# MAGIC %md  残りのパイプラインについても同様にみていきましょう。

# COMMAND ----------

# DBTITLE 1,Descriptionパイプラインの検索
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# COMMAND ----------

# DBTITLE 1,Categorieパイプラインの検索
# ensure top-level environment sees class definition (which must be included in this notebook)
m = __import__('__main__')
setattr(m, 'CategoryFlattener', CategoryFlattener)

# retrieve pipeline from registry
retrieved_category_pipeline = mlflow.spark.load_model(
    model_uri='models:/category_fitted_pipeline/None'
    )

# COMMAND ----------

# MAGIC %md それでは、製品の機能を生成してみましょう。 まず始めに、パイプラインで作成されることを想定したスキーマを持つ空のデータフレームを作成します。これを対象となるデスティネーションに追加して、後でマージ操作を実行できるようなスキーマを持つDeltaLakeテーブルを*lay down*します。 CREATE TABLE IF NOT EXISTS*ステートメントを使用する代わりに、このようにしなければなりません。 このようにしてDeltaLakeテーブルを作成することで、マージの対象となるスキーマを確保することができますが、そのようなスキーマが既に存在している場合には、データを変更することはありません。

# COMMAND ----------

# DBTITLE 1,(必要に応じて)Title特徴量のデルタレイクを作成する
# retreive an empty dataframe
titles_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','title')
  )

# transform empty dataframe to get transformed structure
title_dummy_features = (
  retrieved_title_pipeline
    .transform(titles_dummy)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
title_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/title_pipeline_features')
  )

# COMMAND ----------

# MAGIC %md これで、実データの更新をスキーマに移すことができます。 このパイプライン作業の一環として、どのタイトルを処理するかを決定するために、変更検出ロジックを適用したことに注目してください。

# COMMAND ----------

# DBTITLE 1,タイトルから特徴量に変換する
# retrieve titles to process for features
titles_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','title')
  )

# generate features
title_features = (
  retrieved_title_pipeline
    .transform(titles_to_process)
    .selectExpr('id','tfidf_norm as features','hash')
  )

# COMMAND ----------

# DBTITLE 1,Title特徴量をマージする
# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/title_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      title_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'hash':'source.hash'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md このパターンを説明文やカテゴリー機能に応用してみましょう。

# COMMAND ----------

# DBTITLE 1,Description特徴量を生成する
# retreive an empty dataframe
descript_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','description')
  )

# transform empty dataframe to get transformed structure
descript_dummy_features = (
  retrieved_descript_pipeline
    .transform(descript_dummy)
    .selectExpr('id','w2v','features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
descript_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/descript_pipeline_features')
  )

# retrieve titles to process for features
descriptions_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','description')
  )

# generate features
description_features = (
  retrieved_descript_pipeline
    .transform(descriptions_to_process)
    .selectExpr('id','w2v','features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/descript_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      description_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'w2v':'source.w2v', 'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
  )

# COMMAND ----------

# DBTITLE 1,バケットごとに分布を見る
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   bucket,
# MAGIC   count(*) as products
# MAGIC FROM DELTA.`/mnt/reviews/gold/descript_pipeline_features`
# MAGIC GROUP BY bucket
# MAGIC ORDER BY bucket

# COMMAND ----------

# DBTITLE 1,Category特徴量を生成する
# retreive an empty dataframe
category_dummy = (
  spark
    .table('reviews.metadata')
    .filter('1==2')
    .select('id','category')
  )

# transform empty dataframe to get transformed structure
category_dummy_features = (
  retrieved_category_pipeline
    .transform(category_dummy)
    .selectExpr('id','category_onehot as features','bucket')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
(
category_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/category_pipeline_features')
  )

# retrieve categories to process for features
category_to_process = (
  spark
    .table('reviews.metadata')
    #.repartition(sc.defaultParallelism * 16)
    #.filter('modified_datetime > {0}'.format(last_modified_datetime)) # filter based on last known modified datetime captured (if we have this)
    .select('id','category')
  )

# generate features
category_features = (
  retrieved_category_pipeline
    .transform(category_to_process)
    .selectExpr('id','category_onehot as features','bucket')
  )

# access reviews table in prep for merge
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/category_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      category_features.alias('source'),
      condition='target.id=source.id'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md *"ストローマン"* アーキテクチャに戻って、以下のコンポーネントを実装しました。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_feature_pipelines7.png" width="700">
# MAGIC 
# MAGIC 次に、ユーザープロファイルに目を向けて、このソリューションの機能エンジニアリング面を締めくくりましょう。

# COMMAND ----------

# MAGIC %md  # ♪ステップ3：ユーザープロファイルの生成
# MAGIC 
# MAGIC ユーザープロファイルを生成するために、パイプラインを組み立てることはしません。なぜなら、この作業はDataFrameの操作でより簡単に実装できるからです。しかし、以前のパイプラインのように、あるサイクルで生成する必要のあるプロファイルの数を制限するために、ある種の変更検出を使用することを検討したいと思うかもしれません。 
# MAGIC 
# MAGIC レビューデータでは、新しいレビューを識別するための *unixReviewTime* フィールドにアクセスできます。ユーザープロファイル生成の問題にどのように取り組むかにもよりますが、新しいレビューを持つユーザーだけにプロファイルを作成することも考えられます。現在の日付に基づいて新しいレビューを検出するコードは、レビュー用にここに含まれていますが、コメントアウトされているので、利用可能なすべてのレビュー（前のノートブックで議論した基準を満たすもの）に基づいてプロファイルを生成しています。

# COMMAND ----------

# DBTITLE 1,生のユーザープロファイルの生成
user_profiles_raw = (
  spark
    .sql('''
      SELECT
        a.reviewerID,
        a.overall as rating,
        b.w2v
      FROM reviews.reviews a
      INNER JOIN DELTA.`/mnt/reviews/gold/descript_pipeline_features` b
        ON a.product_id=b.id
      --INNER JOIN (  -- logic to identify reviewers with new relevant reviews in last 1 day
      --  SELECT DISTINCT
      --    reviewerID
      --  FROM reviews.reviews
      --  WHERE 
      --      unixReviewTime >= unix_timestamp( date_sub(current_date(), 1) ) AND
      --      overall >= 4 AND
      --      product_id Is Not Null
      --  ) c
      --  ON a.reviewerID=c.reviewerID
      WHERE a.overall >= 4 AND a.product_id Is Not Null
    ''')
  .groupBy('reviewerID')
    .agg(
      Summarizer.mean(col('w2v'), weightCol=col('rating')).alias('w2v') 
      )
  )

# COMMAND ----------

# MAGIC %md ここで重要なのは、Normalizer トランスフォームはトレーニングに依存していないということです。そのため、再利用のために永続化することを気にすることなく、必要に応じて作成することができます。

# COMMAND ----------

# DBTITLE 1,プロファイルスコアの正規化
user_profiles_norm = (
  Normalizer(inputCol='w2v', outputCol='features', p=2.0)
    .transform(user_profiles_raw)
   )

# COMMAND ----------

# MAGIC %md ノーマライザーとは異なり、プロファイルをバケットに割り当てるために使用されるクラスタリングモデルは、事前のトレーニングに依存します。ユーザープロファイル作成の最初のステップでアクセスした説明の特徴に関連するモデルを取得することが重要です。 ここでは、パイプラインから直接そのモデルにアクセスし、そこに至るまでの変換を省略します。

# COMMAND ----------

# DBTITLE 1,バケットへのプロファイルの割り当て
# retrieve pipeline from registry
retrieved_descript_pipeline = mlflow.spark.load_model(
    model_uri='models:/descript_fitted_pipeline/None'
    )

# assign profiles to clusters/buckets
user_profiles = (
  retrieved_descript_pipeline.stages[-1].transform(user_profiles_norm)
  )

# COMMAND ----------

# MAGIC %md これで、ユーザープロファイルとそのバケット/クラスタの割り当てを永続化することができます。

# COMMAND ----------

# DBTITLE 1,プロファイルの永続化
# retreive an empty dataframe
user_profiles_dummy_features = (
  user_profiles
    .filter('1==2')
    .select('reviewerID','features','bucket')
  )

# transform empty dataframe to get transformed structure
(
user_profiles_dummy_features
  .write
  .format('delta')
  .mode('append')
  .save('/mnt/reviews/gold/user_profile_pipeline_features')
  )

# persist the empty dataframe to laydown the schema (if not already in place)
target = DeltaTable.forPath(spark, '/mnt/reviews/gold/user_profile_pipeline_features')

# perform merge on target table
( target.alias('target')
    .merge(
      user_profiles.alias('source'),
      condition='target.reviewerID=source.reviewerID'
      )
    .whenMatchedUpdate(set={'features':'source.features', 'bucket':'source.bucket'})
    .whenNotMatchedInsertAll()
  ).execute()

# display features data
display(
  spark.table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
  )

# COMMAND ----------

# MAGIC %md アーキテクチャのうち、「Featuring Engineering」の部分が完成しました。

# COMMAND ----------

# MAGIC %md # Step 4a: 製品ベースのレコメンデーションの作成
# MAGIC 
# MAGIC ここまでの数ステップで紹介したパイプラインを利用して機能を実装し、ジョブを最新の状態に保つことができたので、今度はアプリケーション層にレコメンデーションをどのように公開するかを考えてみましょう。レコメンデーションの取得に関するパフォーマンスは不可欠ですが、ポートフォリオには非常に多くの製品が含まれているため、レコメンデーションを事前に生成してキャッシュすることは、時間とコストがかかる可能性があります。
# MAGIC 
# MAGIC 適切なソリューションは、これらすべての懸念事項のバランスを取る必要があり、このノートの冒頭で述べたように、多くの関係者の意見に依存します。しかし、バッチ処理で生成されたものであれ、再利用を可能にするオンザフライのものであれ、ある程度のキャッシングは多くのソリューションの構成要素になると想像しています。レコメンデーション データのキャッシングおよび提供のための一般的なターゲットには次のようなものがあり、すべて Databricks で書き込むことができます：</p> * [Redis]()
# MAGIC 
# MAGIC * [Redis](https://docs.databricks.com/data/data-sources/redis.html)
# MAGIC * [MongoDB](https://docs.databricks.com/data/data-sources/mongodb.html)
# MAGIC * [CouchBase](https://docs.databricks.com/data/data-sources/couchbase.html)
# MAGIC * [Azure CosmosDB](https://docs.databricks.com/data/data-sources/azure/cosmosdb-connector.html)
# MAGIC * AWS DocumentDB: MongoDBに関する情報を参照するか、オーケストレーションに[AWS Glue](https://aws.amazon.com/about-aws/whats-new/2020/04/aws-glue-now-supports-reading-from-amazon-documentdb-and-mongodb-tables/)の使用を検討する。
# MAGIC 
# MAGIC 今回のソリューションでは、レコメンデーションを事前に生成し、その結果をシンプルに表示するために必要なロジックに焦点を当てます。 上記のリンクで提供されているコードサンプルを使えば、組み立てられたレコメンデーションをDataFramesから目的のキャッシングプラットフォームに移すことができるはずです。

# COMMAND ----------

# MAGIC %md まず最初に、どの製品に対してレコメンドを行うかを決定します。 ここでは、「ホーム＆キッチン」カテゴリーの製品10,000点を対象とします。 実際には、製品のバックログを効率的に処理したり、製品ポートフォリオの変化に対応したりするために、処理する製品の数を増やしたり減らしたりする必要があるかもしれません。
# MAGIC 
# MAGIC ここでは、推奨する製品を「a製品」と呼び、潜在的な推奨製品を構成する製品を「b製品」と呼びます。 b製品は、同じ上位製品カテゴリー（例：*ホーム＆キッチン*）に属する製品に限定していますが、これはお客様のニーズに合ったアプローチであるかどうかは別です。

# COMMAND ----------

# DBTITLE 1,提案する製品の決定
## parallelize downstream work
#spark.conf.set('spark.sql.shuffle.partitions', sc.defaultParallelism * 10)

# products we ae making recommendations for
product_a =  (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_features`')
    .filter("bucket = 'Home & Kitchen'")
    .limit(10000) # we would typically have some logic here to identify products that we need to generate recommendations for
    .selectExpr('id as id_a')
  ).cache()

# recommendation candidates to consider
product_b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_features`')
    .filter("bucket = 'Home & Kitchen'")
    .selectExpr('id as id_b')
  ).cache()

# COMMAND ----------

# MAGIC %md それでは、コンテンツ由来の特徴を使って類似度を計算してみましょう。 このプロセスは、上述の特徴量生成のステップとは別のものとして想定していますが、このステップを繰り返さないように、以下のコードではすでに検索されたパイプラインを利用します。
# MAGIC 
# MAGIC 類似度のスコアと製品間の距離（LSHルックアップ）にいくつかの制限を設けていることに注意してください。 これらの制限は、最終的なレコメンデーション・スコアリングの計算に向かうデータ量を減らすために実装されています。 このような制限を設けるかどうかは、お客様のデータやビジネスニーズを考慮して決定してください。

# COMMAND ----------

# DBTITLE 1,相似関数の定義
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for jaccard similarity calc
# MAGIC // from https://github.com/karlhigley/spark-neighbors/blob/master/src/main/scala/com/github/karlhigley/spark/neighbors/linalg/DistanceMeasure.scala
# MAGIC val jaccard_similarity = udf { (v1: SparseVector, v2: SparseVector) =>
# MAGIC   val indices1 = v1.indices.toSet
# MAGIC   val indices2 = v2.indices.toSet
# MAGIC   val intersection = indices1.intersect(indices2)
# MAGIC   val union = indices1.union(indices2)
# MAGIC   intersection.size.toDouble / union.size.toDouble
# MAGIC }
# MAGIC spark.udf.register("jaccard_similarity", jaccard_similarity)
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,カテゴリの類似性の算出
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .withColumn('features_a', col('features'))
    .withColumn('id_a',col('id'))
    .drop('features','id')
    )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/category_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .withColumn('features_b', col('features'))
    .withColumn('id_b',col('id'))
    .drop('features','id')
    )

# similarity results
category_similarity = (
  a.crossJoin(b)
    .withColumn('similarity', expr('jaccard_similarity(features_a, features_b)'))
    .select('id_a', 'id_b', 'similarity')
    )

display(category_similarity)

# COMMAND ----------

# DBTITLE 1,タイトルの類似性の計算
# categories for products for which we wish to make recommendations
a = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
    .selectExpr('id as id_a','features as tfidf_norm')
  )

# categories for products which will be considered for recommendations
b = (
  spark
    .table('DELTA.`/mnt/reviews/gold/title_pipeline_features`')
    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
    .selectExpr('id as id_b','features as tfidf_norm')
  )

# retrieve fitted LSH model from pipeline
fitted_lsh = retrieved_title_pipeline.stages[-1]

# similarity results
title_similarity = (
  fitted_lsh.approxSimilarityJoin(
      a,  
      b,
      threshold = 1.4, # this is pretty high so that we can be more inclusive.  Setting it at or above sqrt(2) will bring back every product
      distCol='distance'
      )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .selectExpr('datasetA.id_a as id_a', 'datasetB.id_b as id_b', 'similarity')
    .filter('similarity > 0.01') # set a lower limit for consideration
    )

display(title_similarity)

# COMMAND ----------

# MAGIC %md **NOTE** 説明文ベースの類似性を計算するコードを掲載していますが、このノートブックの最後にある最終的な計算には使用していません。

# COMMAND ----------

# DBTITLE 1,説明文の類似性の算出
## categories for products for which we wish to make recommendations
#a = (
#  spark
#    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
#    .join(product_a, on=[col('id')==product_a.id_a], how='left_semi')
#    .selectExpr('id as id_a', 'features as features_a', 'bucket')
#  )
#
## categories for products which will be considered for recommendations
#b = (
#  spark
#    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
#    .join(product_b, on=[col('id')==product_b.id_b], how='left_semi')
#    .selectExpr('id as id_b', 'features as features_b', 'bucket')
#  )
#
## caculate similarities
#description_similarity = (
#  a  
#    .hint('skew','bucket')
#    .join(b, on=[a.bucket==b.bucket], how='inner')
#    .withColumn('similarity', expr('euclidean_similarity(features_a, features_b)'))
#    .select('id_a','id_b','similarity')
#    .filter('similarity > 0.01') # set a lower limit for consideration
#  )
#
#display(description_similarity)

# COMMAND ----------

# MAGIC %md タイトルとカテゴリーのデータを利用して類似性スコアを算出し、これらのスコアを組み合わせて最終スコアを作成します。 上記のステップでは、バケットを使って比較の数を制限しましたが、ここで返すことのできる商品はまだたくさんあります。 シンプルな関数を使って、上位N個の製品に制限しています。
# MAGIC 
# MAGIC **注** 行番号関数とフィルタ制約を使用して同様のことを行うこともできますが、この状況では関数の方が少しうまく機能することがわかりました。

# COMMAND ----------

# DBTITLE 1,レコメンデーションの生成
# function to get top N product b's for a given product a
def get_top_products( data ):
  '''the incoming dataset is expected to have the following structure: id_a, id_b, score'''  
  
  rows_to_return = 11 # limit to top 10 products (+1 for self)
  
  return data.sort_values(by=['score'], ascending=False).iloc[0:rows_to_return] # might be faster ways to implement this sort/trunc

# combine similarity scores and get top N highest scoring products
recommendations = (
  category_similarity
    .join(title_similarity, on=['id_a', 'id_b'], how='inner')
    .withColumn('score', category_similarity.similarity * title_similarity.similarity)
    .select(category_similarity.id_a, category_similarity.id_b, 'score')  
    .groupBy('id_a')
      .applyInPandas(
        get_top_products, 
        schema='''
          id_a long,
          id_b long,
          score double
          ''')
    )

display(recommendations)

# COMMAND ----------

# DBTITLE 1,キャッシュされたデータセットのクリーンアップ
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()

# COMMAND ----------

# MAGIC %md # Step 4b: プロファイルベースのレコメンデーションの作成
# MAGIC 
# MAGIC コンテンツのみのレコメンダーと同様に、レコメンデーションを事前に計算してキャッシュすべきか、それとも動的に計算すべきかを決定するには多くの要因があります。 ここでは、コンテンツのみのレコメンダーと同様に、レコメンデーションを事前に計算してキャッシュするか、それとも動的に計算するかを決定するための要素が数多くあります。
# MAGIC 
# MAGIC ここでは、レコメンデーションの生成を限られた数のユーザーに限定し、レコメンデーションの対象をスコアの高い上位25項目に限定します。

# COMMAND ----------

# DBTITLE 1,類似性関数の定義
# MAGIC %scala
# MAGIC 
# MAGIC import math._
# MAGIC import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}
# MAGIC 
# MAGIC // function for euclidean distance derived similarity
# MAGIC val euclidean_similarity = udf { (v1: Vector, v2: Vector) =>
# MAGIC   val distance = sqrt(Vectors.sqdist(v1, v2))
# MAGIC   val rawSimilarity = 1 / (1+distance)
# MAGIC   val minScore = 1 / (1+sqrt(2))
# MAGIC   (rawSimilarity - minScore)/(1 - minScore)
# MAGIC }
# MAGIC spark.udf.register("euclidean_similarity", euclidean_similarity)

# COMMAND ----------

# DBTITLE 1,レコメンデーション用データセットの取得
products = (
  spark
    .table('DELTA.`/mnt/reviews/gold/descript_pipeline_features`')
    .withColumnRenamed('features', 'features_b')
    .cache()
    )
products.count() # force caching to complete

user_profiles = (
  spark
    .table('DELTA.`/mnt/reviews/gold/user_profile_pipeline_features`')
    .withColumnRenamed('features', 'features_a')
    #.sample(False, 0.00005)
    ).cache()

# see how profiles distributed between buckets
display(
  user_profiles
    .groupBy('bucket')
      .agg(count('*'))
  )

# COMMAND ----------

# DBTITLE 1,レコメンデーションの生成
# make recommendations for sampled reviewers
recommendations = (
  products
    .hint('skew','bucket') # hint to ensure join is balanced
    .join( 
      user_profiles, 
      on='bucket', 
      how='inner'
      ) # join products to profiles on buckets
    .withColumn('score', expr('euclidean_similarity(features_a, features_b)')) # calculate similarity
    .withColumn('seq', expr('row_number() OVER(PARTITION BY reviewerID ORDER BY score DESC)'))
    .filter('seq <= 25')
    .selectExpr('reviewerID', 'id as product_id', 'seq')
    )

display(
  recommendations
  )

# COMMAND ----------

# DBTITLE 1,キャッシュされたデータセットのクリーンアップ
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
