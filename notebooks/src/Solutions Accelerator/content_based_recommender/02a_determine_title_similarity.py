# Databricks notebook source
# MAGIC %md # イントロダクション
# MAGIC 
# MAGIC このノートブックの目的は、製品間の類似性を計算するために、製品タイトルからどのように特徴を抽出できるかを調べることである。(これらの類似性は、**関連製品**を推奨するための基礎として使用されます。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/reviews_recommendations.png" width="600">
# MAGIC 
# MAGIC このノートブックは **Databricks ML 7.3+ クラスタ** で実行する必要があります。

# COMMAND ----------

# MAGIC %md **注意** このノートブックを実行するクラスタは、NLTK WordNetコーパスとAveraged Perceptron Taggerをインストールする[cluster-scoped initialization script](https://docs.databricks.com/clusters/init-scripts.html?_ga=2.158476346.1681231596.1602511918-995336416.1592410145#cluster-scoped-init-script-locations)を使用して作成する必要があります。 以下のセルを使用してこのようなスクリプトを生成することができますが、スクリプトに依存するコードを実行する前に、このスクリプトをクラスターに関連付ける必要があります。

# COMMAND ----------

# DBTITLE 1,Cluster Init Scriptの生成
dbutils.fs.mkdirs('dbfs:/FileStore/databricks/scripts/')

dbutils.fs.put(
  '/FileStore/databricks/scripts/install-nltk-downloads.sh',
  '''#!/bin/bash\n/databricks/python/bin/python -m nltk.downloader wordnet\n/databricks/python/bin/python -m nltk.downloader averaged_perceptron_tagger''', 
  True
  )

# alternatively, you could install all NLTK elements with: python -m nltk.downloader all

# show script content
print(
  dbutils.fs.head('dbfs:/FileStore/databricks/scripts/install-nltk-downloads.sh')
  )

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
import nltk

import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, lit, count, countDistinct, col, array_join, expr, monotonically_increasing_id, explode

from typing import Iterator

from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, Normalizer, RegexTokenizer, BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vector, Vectors

from pyspark.sql import DataFrame

import mlflow.spark

import shutil 

# COMMAND ----------

# MAGIC %md # Step 1: タイトルデータの準備
# MAGIC 
# MAGIC 類似性の高い製品を推奨することを目的としている場合、製品名に基づいてそのような製品を特定することがあります。 この情報は、このデータセットの*タイトル*フィールドに取り込まれます。

# COMMAND ----------

# DBTITLE 1,タイトルの抽出
# retrieve titles for each product
titles = (
  spark
    .table('reviews.metadata')
    .filter('title Is Not Null')
    .select('id', 'asin', 'title')
  )

num_of_titles = titles.count()
num_of_titles

# COMMAND ----------

# present titles
display(
  titles
  )

# COMMAND ----------

# MAGIC %md タイトル間の類似性を比較するためには、非常に簡単な単語ベースの比較を採用することができます。ここでは、各単語がタイトル内での出現率と全タイトルでの出現率に基づいて重み付けされます。 これらの重みは、多くの場合、*term-frequency - inverse document frequency* (*TF-IDF*) スコアを使用して計算されます。
# MAGIC 
# MAGIC 
# MAGIC TF-IDF スコアを計算する最初のステップとして、タイトルの中の単語を分割して、一貫したケースに移動させる必要があります。 これは、[RegexTokenizer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RegexTokenizer)を使って行います。
# MAGIC 
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_coasters.jpg' width='150'>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_tokenization2.png' width='1100'>
# MAGIC 
# MAGIC Sparkにはホワイトスペースでテキストを分割するシンプルな[Tokenizer](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Tokenizer)も用意されていますが、今回の*title*フィールドのように複雑なテキストデータの場合は、RegexTokenizerを使うことで、迷子の句読点などをうまく処理することができます。

# COMMAND ----------

# DBTITLE 1,タイトルから単語を抽出する
# split titles into words
tokenizer = RegexTokenizer(
    minTokenLength=2, 
    pattern='[^\w\d]',  # split on "not a word and not a digit"
    inputCol='title', 
    outputCol='words'
    )

title_words = tokenizer.transform(titles)

display(title_words)

# COMMAND ----------

# MAGIC %md 単語が分割されたので、次に、単数形と複数形、未来形、現在形、過去形の動詞などの一般的な単語のバリエーションをどう処理するかを考えます。
# MAGIC 
# MAGIC このような場合、[ステミング](https://en.wikipedia.org/wiki/Stemming) という手法があります。 ステム処理では、一般的な単語の接尾辞を削除して、単語をそのルート (ステム) に切り詰めます。 効果的ではありますが、ステミングには、単語がどのように使用されているか、また、非標準的な形 態を持つ単語 (例: *man vs. men*) がどのように関連しているかについての知識がありません。[レマタイゼーション](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) と呼ばれる少し洗練された技術を使用すると、単語の形をよりよく標準化できます。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_coasters.jpg' width='150'><img src='https://brysmiwasb.blob.core.windows.net/demos/images/reviews_lemmatization2.png' width='1100'>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 単語を*lemmata*（*lemma*の複数形）に変換するために使用できるライブラリは様々あります。 [NLTK](https://buildmedia.readthedocs.org/media/pdf/nltk/latest/nltk.pdf)はこれらの中でも特に人気があり、ほとんどのDatabricks MLの実行環境にはあらかじめインストールされています。 しかし、NLTKを使ってレマタイゼーションを行うためには、タグ付きコーパスと品詞予測器をクラスタ内の各ワーカーノードにインストールする必要があります。 これは、このノートブックの冒頭で説明したように、initスクリプトでクラスタを構成することによって行われます。
# MAGIC 
# MAGIC ここでは、[WordNetコーパス](https://www.nltk.org/howto/wordnet.html)を使用して、単語のコンテキストを提供しています。 この文脈を利用して、単語を標準化するだけでなく、形容詞、名詞、動詞、副詞など、一般的に情報量の多い品詞として使われていない単語を排除します。 代替の*corpora*（*corpus*の複数形）はNLTKで[ダウンロード](http://www.nltk.org/howto/corpus.html)することができますが、異なる結果が得られる可能性があります。
# MAGIC 
# MAGIC **注** 私はpandas UDFを使用して、*iterator of series to iterator of series*タイプのレマタイズロジックを実装しています。 これは[New style of pandas UDF (in Spark 3.0)](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html)で、高価な初期化が行われるシナリオでは便利です。

# COMMAND ----------

# DBTITLE 1,単語を標準化させる
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
title_lemmata = (
  title_words
    .withColumn(
      'lemmata',
      lemmatize_words('words') 
      )
    .select('id','asin', 'title', 'lemmata')
    ).cache()

display(title_lemmata)

# COMMAND ----------

# MAGIC %md # Step 2: TF-IDFスコアを算出する
# MAGIC 
# MAGIC データの準備ができたら、TF-IDF計算の「単語の頻度」の部分の計算を進めます。 ここでは、タイトル内の単語の出現数を単純に数えてみましょう。 タイトルは一般的に簡潔であるため、ほとんどの単語は1つのタイトルに1回しか現れないことが予想されます。 類似性の比較に役立たないような珍しい単語をカウントしないように、カウントする単語数を全タイトルの上位262,144語に制限します。 これはSparkのワードカウンターのデフォルト設定ですが、制限があることを明確にするためにコードで明示的にこの値を割り当てています。
# MAGIC 
# MAGIC 単語をカウントするには、2つの基本的なオプションがあります。 [CountVectorizer](https://spark.apache.org/docs/latest/ml-features.html#countvectorizer)はブルートフォース方式でカウントを行いますが、これは小さいテキストコンテンツには有効です。 次のノートでは、代替の用語頻度変換器であるHashTFを使用します。

# COMMAND ----------

# DBTITLE 1,タイトルの中に単語頻出をカウントする
# count word occurences
title_tf = (
  CountVectorizer(
    inputCol='lemmata', 
    outputCol='tf',
    vocabSize=262144  # top n words to consider
    )
    .fit(title_lemmata)
    .transform(title_lemmata)
    )

display(title_tf.select('id','asin','lemmata','tf'))

# COMMAND ----------

# MAGIC %md ここで、タイトルに含まれる単語の「逆文書頻度」（IDF）を計算します。ある単語がタイトル間で頻繁に使用されるようになると、IDF スコアは対数的に減少し、その単語が持つ差別化情報が少なくなっていきます。 生のIDFスコアは通常、TFスコアと掛け合わされ、目的のTF-IDFスコアが生成されます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/review_tfidf.png" width="400">
# MAGIC 
# MAGIC これらはすべて[IDFトランスフォーム](https://spark.apache.org/docs/latest/ml-features.html#tf-idf)で対応しています。

# COMMAND ----------

# DBTITLE 1,TF-IDFスコアを算出
# calculate tf-idf scores
title_tfidf = (
  IDF(inputCol='tf', outputCol='tfidf')
    .fit(title_tf)
    .transform(title_tf)
  )

display(
  title_tfidf.select('id','asin','lemmata','tfidf')
  )

# COMMAND ----------

# MAGIC %md 変換によって返されるTF-IDFスコアは正規化されていません。 距離計算に基づく類似性計算では、頻繁にL2正規化を適用します。 これは、TF-IDFスコアに[Normalizer](https://spark.apache.org/docs/latest/ml-features.html#normalizer)変換を適用することで対処されます。

# COMMAND ----------

# DBTITLE 1,TF-IDF値を正規化する
# apply normalizer
title_tfidf_norm = (
  Normalizer(inputCol='tfidf', outputCol='tfidf_norm', p=2.0)
    .transform(title_tfidf)
  )

display(title_tfidf_norm.select('id','asin','lemmata','tfidf','tfidf_norm'))

# COMMAND ----------

# MAGIC %md # Step 3: 似たようなタイトルの商品を探す
# MAGIC 
# MAGIC タイトル間の類似性を計算できる機能ができました。ブルートフォース（総当り）方式では、1,180万本のタイトルをそれぞれ比較し、約70兆回の比較を行うことになります。 これでは、たとえ分散型のシステムであっても、経済的に成り立ちません。 その代わりに、比較対象を類似している可能性の高い製品に限定するための近道を見つける必要があります。
# MAGIC 
# MAGIC コラボレーション・フィルタリング・ノートでは、この問題に取り組むための1つのアプローチとして、[Local Sensitive Hashing](https://spark.apache.org/docs/latest/ml-features.html#locality-sensitive-hashing)を検討しました。 (その手法を応用して、似たようなタイトルを探すことができます。)

# COMMAND ----------

# DBTITLE 1,LSHをタイトルに適用する
# configure lsh
bucket_length = 0.0001
lsh_tables = 5

# fit the algorithm to the dataset 
fitted_lsh = (
  BucketedRandomProjectionLSH(
    inputCol = 'tfidf_norm', 
    outputCol = 'hash', 
    numHashTables = lsh_tables, 
    bucketLength = bucket_length
    ).fit(title_tfidf_norm)
  )

# assign LSH buckets to users
hashed_vectors = (
  fitted_lsh
    .transform(title_tfidf_norm)
    ).cache()

display(
  hashed_vectors.select('id','asin','title','tfidf_norm','hash')
  )

# COMMAND ----------

# MAGIC %md LSHを使うことで、タイトルをほぼ同じ値のバケツに素早く分類することができました。 この技術は完璧ではありませんが、サンプルの製品を見ればわかるように、類似した製品を合理的に見つけることができます。

# COMMAND ----------

# DBTITLE 1,サンプル製品の抽出情報
# retrieve data for example product
sample_product = hashed_vectors.filter('asin==\'B01G6M7CLK\'') 
                                       
display(
  sample_product.select('id','asin','title','tfidf_norm','hash')
  )                                     

# COMMAND ----------

# DBTITLE 1,100種類の類似製品を検索 
number_of_titles = 100

# retrieve n nearest customers 
similar_k_titles = (
  fitted_lsh.approxNearestNeighbors(
    hashed_vectors, 
    sample_product.collect()[0]['tfidf_norm'], 
    number_of_titles, 
    distCol='distance'
    )
    .withColumn('raw_sim', expr('1/(1+distance)'))
    .withColumn('min_score', expr('1/(1+sqrt(2))'))
    .withColumn('similarity', expr('(raw_sim - min_score)/(1-min_score)'))
    .select('id', 'asin', 'title', 'distance', 'similarity')
  )
  
display(similar_k_titles)

# COMMAND ----------

# MAGIC %md データを見てみると、似たようなタイトルの商品がかなりあることがわかります。私たちのアプローチがどの程度完全なものかを知るために、サンプル製品が属するカテゴリー内の製品の幅を調べるかもしれませんが、いずれにしても、私たちの評価には避けがたい主観性が存在しています。これは、製品の評価や購入履歴など、比較すべき基準となる真実がないレコメンデーションを検討する際に、非常によく見られる課題です。ここでできることは、設定を変更して適度に良いと思われるレコメンデーションのセットにたどり着き、実際のお客様で限定的なテストを行い、私たちの提案に対するお客様の反応を確認することです。

# COMMAND ----------

# DBTITLE 1,キャッシュの削除
def list_cached_dataframes():
    return [(k,v) for (k,v) in [(k,v) for (k, v) in globals().items() if isinstance(v, DataFrame)] if v.is_cached]
  
for name, obj in list_cached_dataframes():
  obj.unpersist()
