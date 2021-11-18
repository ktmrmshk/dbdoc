# Databricks notebook source
# MAGIC %md
# MAGIC # ESG - レポート
# MAGIC 
# MAGIC 金融の未来は、社会的責任、環境スチュワードシップ、企業倫理と密接に関わっています。競争力を維持するために、金融サービス機関（FSI）は、自社の**環境**、**社会**、**ガバナンス**（ESG）パフォーマンスに関するより多くの情報を開示するようになってきています。企業や事業への投資が持続可能性と社会的影響をよりよく理解し、定量化することで、金融機関はレピュテーションリスクを軽減し、顧客や株主との信頼関係を維持することができます。データブリックスでは、顧客からESGがCスイートの優先事項になっているとの声を聞くことが増えています。これは利他主義だけではなく、経済学的な理由もあります。このデモでは、NLPテクニックとグラフ分析を組み合わせることで、持続可能な金融への斬新なアプローチを提供します。: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). 今回のデモでは、NLP手法とグラフ分析を組み合わせて、戦略的なESGの主要な取り組みを抽出し、グローバル市場における企業の関係性や市場リスク計算への影響を学ぶことで、持続可能な金融への斬新なアプローチをご提案します。
# MAGIC 
# MAGIC ---
# MAGIC + <a href="https://databricks.com/notebooks/esg_notebooks/01_esg_report.html">STAGE1</a>: NLPを使用して重要なESGイニシアチブをPDFレポートから抽出
# MAGIC + <a href="https://databricks.com/notebooks/esg_notebooks/02_esg_scoring.html">STAGE2</a>: グラフ分析を用いたESGスコアリングを新しいアプローチで実施
# MAGIC + <a href="https://databricks.com/notebooks/esg_notebooks/03_esg_market.html">STAGE3</a>: 市場リスクの計算にESGを適用
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 内容
# MAGIC 金融機関は現在、環境、社会、ガバナンス戦略に関する情報開示を株主から求められることが多くなってきています。通常、企業は毎年ウェブサイト上で PDF 文書の形で公表しており、従業員、顧客、顧客をどのように評価しているか、社会に積極的に貢献しているか、あるいは炭素排出量をどのように削減しているか（あるいは削減にコミットしているか）など、複数のテーマにまたがって ESG の主要な取り組みを伝えています。これらの報告書は通常、第三者機関（[msci](https://www.msci.com/esg-ratings)や[csrhub](https://www.csrhub.com/)など）が作成したもので、業界を超えて統合され、ベンチマークされてESGメトリクスが作成されます。このノートでは、トップレベルの金融サービス機関の40以上のESGレポートにプログラムでアクセスし、さまざまなトピックにわたる主要なESGイニシアチブを学びたいと考えています。
# MAGIC 
# MAGIC ### Dependencies
# MAGIC 以下のように、Sparkクラスタ全体で利用できるようにする必要がある複数のサードパーティ製ライブラリを使用しています。**注** 次のセルでは、MLランタイムを利用していないDatabricksクラスタでこのノートブックを実行していることを前提としています。 MLランタイムを使用している場合は、以下の[別ステップ](https://docs.databricks.com/libraries.html#workspace-library)に従ってください。

# COMMAND ----------

# DBTITLE 1,Install needed libraries
dbutils.library.installPyPI('PyPDF2')
dbutils.library.installPyPI('spacy')
dbutils.library.installPyPI('gensim')
dbutils.library.installPyPI('wordcloud')
dbutils.library.installPyPI('mlflow')
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install PyPDF2

# COMMAND ----------

# MAGIC %pip install spacy

# COMMAND ----------

# MAGIC %pip install gensim

# COMMAND ----------

# MAGIC %pip install wordcloud

# COMMAND ----------

# DBTITLE 1,ESG ワークスペース作成
# MAGIC %sql
# MAGIC CREATE DATABASE esg;

# COMMAND ----------

# DBTITLE 1,ライブラリー群のインポート
import warnings
import requests
import PyPDF2
import io
import re
import string
import pandas as pd
import numpy as np
import gensim
import spacy
from spacy import displacy
import uuid
import os
import json

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans

from pyspark.sql.functions import col, pandas_udf, PandasUDFType
from pyspark.sql import functions as F
from pyspark.sql.functions import udf

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

warnings.simplefilter("ignore", DeprecationWarning)
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1:  ESGレポートを抽出
# MAGIC このセクションでは、トップレベルの FSI から公開されている ESG レポートを手動で検索しています。今日の時点では、これらのレポートを企業や業界全体で統合してくれる中央レポジトリを私は知らないので、特定の企業のESGレポートをダウンロードするためのすべてのURLを提供しなければなりません。すべてのPDFが異なる形式のものである可能性があるため、レポートを明確に定義された文に統合し、`spacy`を使って文法的に妥当な文を抽出することに多くの時間を費やさなければなりません。私たちのデータセットは比較的小さいのですが、スペイシーモデルをロードして実行するのはコストのかかるプロセスです。当社では、`pandasUDF`パラダイムを活用してモデルを一度だけロードすることで、すべての投資先のESGドキュメントの大規模なコレクションに対してプロセスを簡単に拡張できるようにしています。

# COMMAND ----------

# DBTITLE 1,ESGレポートのURLを定義する
esg_urls_rows = [
  ['barclays', 'https://home.barclays/content/dam/home-barclays/documents/citizenship/ESG/Barclays-PLC-ESG-Report-2019.pdf'],
  ['jp morgan chase', 'https://impact.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-cr-esg-report-2019.pdf'],
  ['morgan stanley', 'https://www.morganstanley.com/pub/content/dam/msdotcom/sustainability/Morgan-Stanley_2019-Sustainability-Report_Final.pdf'],
  ['goldman sachs', 'https://www.goldmansachs.com/what-we-do/sustainable-finance/documents/reports/2019-sustainability-report.pdf'],
  ['hsbc', 'https://www.hsbc.com/-/files/hsbc/our-approach/measuring-our-impact/pdfs/190408-esg-update-april-2019-eng.pdf'],
  ['citi', 'https://www.citigroup.com/citi/about/esg/download/2019/Global-ESG-Report-2019.pdf'],
  ['td bank', 'https://www.td.com/document/PDF/corporateresponsibility/2018-ESG-Report.pdf'],
  ['bank of america', 'https://about.bankofamerica.com/assets/pdf/Bank-of-America-2017-ESG-Performance-Data-Summary.pdf'],
  ['rbc', 'https://www.rbc.com/community-social-impact/_assets-custom/pdf/2019-ESG-Report.PDF'],
  ['macquarie', 'https://www.macquarie.com/assets/macq/investor/reports/2020/sections/Macquarie-Group-FY20-ESG.pdf'],
  ['lloyds', 'https://www.lloydsbankinggroup.com/globalassets/documents/investors/2020/2020feb20_lbg_esg_approach.pdf'],
  ['santander', 'https://www.santander.co.uk/assets/s3fs-public/documents/2019_santander_esg_supplement.pdf'],
  ['bluebay', 'https://www.bluebay.com/globalassets/documents/bluebay-annual-esg-investment-report-2018.pdf'],
  ['lasalle', 'https://www.lasalle.com/documents/ESG_Policy_2019.pdf'],
  ['riverstone', 'https://www.riverstonellc.com/media/1196/riverstone_esg_report.pdf'],
  ['aberdeen standard', 'https://www.standardlifeinvestments.com/RI_Report.pdf'],
  ['apollo', 'https://www.apollo.com/~/media/Files/A/Apollo-V2/documents/apollo-2018-esg-summary-annual-report.pdf'],
  ['bmogan', 'https://www.bmogam.com/gb-en/intermediary/wp-content/uploads/2019/02/cm16148-esg-profile-and-impact-report-2018_v33_digital.pdf'],
  ['vanguard', 'https://personal.vanguard.com/pdf/ISGESG.pdf'],
  ['ruffer', 'https://www.ruffer.co.uk/-/media/Ruffer-Website/Files/Downloads/ESG/2018_Ruffer_report_on_ESG.pdf'],
  ['northern trust', 'https://cdn.northerntrust.com/pws/nt/documents/fact-sheets/mutual-funds/institutional/annual-stewardship-report.pdf'],
  ['hermes investments', 'https://www.hermes-investment.com/ukw/wp-content/uploads/sites/80/2017/09/Hermes-Global-Equities-ESG-Dashboard-Overview_NB.pdf'],
  ['abri capital', 'http://www.abris-capital.com/sites/default/files/Abris%20ESG%20Report%202018.pdf'],
  ['schroders', 'https://www.schroders.com/en/sysglobalassets/digital/insights/2019/pdfs/sustainability/sustainable-investment-report/sustainable-investment-report-q2-2019.pdf'],
  ['lazard', 'https://www.lazardassetmanagement.com/docs/-m0-/54142/LazardESGIntegrationReport_en.pdf'],
  ['credit suisse', 'https://www.credit-suisse.com/pwp/am/downloads/marketing/br_esg_capabilities_uk_csam_en.pdf'],
  ['coller capital', 'https://www.collercapital.com/sites/default/files/Coller%20Capital%20ESG%20Report%202019-Digital%20copy.pdf'],
  ['cinven', 'https://www.cinven.com/media/2086/81-cinven-esg-policy.pdf'],
  ['warburg pircus', 'https://www.warburgpincus.com/content/uploads/2019/07/Warburg-Pincus-ESG-Brochure.pdf'],
  ['exponent', 'https://www.exponentpe.com/sites/default/files/2020-01/Exponent%20ESG%20Report%202018.pdf'],
  ['silverfleet capital', 'https://www.silverfleetcapital.com/media-centre/silverfleet-esg-report-2020.pdf'],
  ['kkr', 'https://www.kkr.com/_files/pdf/KKR_2018_ESG_Impact_and_Citizenship_Report.pdf'],
  ['cerberus', 'https://www.cerberus.com/media/2019/07/Cerberus-2018-ESG-Report_FINAL_WEB.pdf'],
  ['standard chartered', 'https://av.sc.com/corp-en/others/2018-sustainability-summary2.pdf'],
]

# ESGレポートURLを含むPandasデータフレームを作成
esg_urls_pd = pd.DataFrame(esg_urls_rows, columns=['company', 'url'])

# 我々は小さなコレクションをクラスタに分散させて...
# ...情報をダウンロードしたりキュレーションしたりするときに並列分散を行います。
esg_urls = spark.createDataFrame(esg_urls_pd).repartition(8)

# COMMAND ----------

# DBTITLE 1,ESGレポートのPDFコンテンツを抽出
@udf('string')
def extract_content(url):
  """
  A simple user define function that, given a url, download PDF text content
  Parse PDF and return plain text version
  """
  try:
    # PDFをバイナリーストリームで取得
    response = requests.get(url)
    open_pdf_file = io.BytesIO(response.content)
    pdf = PyPDF2.PdfFileReader(open_pdf_file)  
    # PDFコンテンツにアクセス
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    # 連結した内容を返す
    return "\n".join(text)
  except:
    return ""
    
# PDFのESGレポートをダウンロード
esg_articles = esg_urls \
  .withColumn('content', extract_content(F.col('url'))) \
  .filter(F.length(F.col('content')) > 0) \
  .cache()

esg_articles.count()
display(esg_articles)

# COMMAND ----------

# DBTITLE 1,ESGレポートのステートメントを抽出
def remove_non_ascii(text):
  printable = set(string.printable)
  return ''.join(filter(lambda x: x in printable, text))

def not_header(line):
  # 改行した行を段落にまとめているので、ヘッダを含まないようにします
  return not line.isupper()

def extract_statements(nlp, text):
  """
  Extracting ESG statements from raw text by removing junk, URLs, etc.
  We group consecutive lines into paragraphs and use spacy to parse sentences.
  """
  
  # ASCII文字を取り除きます
  text = remove_non_ascii(text)
  
  lines = []
  prev = ""
  for line in text.split('\n'):
    # テキストが分割される可能性のある連続した行を集約する
    # 次の行がスペースで始まるか、前の行がドットで終わらない場合に限ります。    
    if(line.startswith(' ') or not prev.endswith('.')):
        prev = prev + ' ' + line
    else:
        # 新しい段落
        lines.append(prev)
        prev = line
        
  # 残りの段落を考慮
  lines.append(prev)

  # 余分なスペースから段落をクレンジングする。不要な文字、 URLなど。
  # ベストエフォートでクレンジング。もっと汎用性の高いクレンジングは検討要。
  sentences = []
  for line in lines:
    
      # ヘッダー番号を削除
      line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
      # 終わりのスペースを削除
      line = line.strip()
      # 単語は行跨ぐことがあります、それらをリンクさせます
      line = re.sub('\s?-\s?', '-', line)
      # 句読点の前のスペースを削除
      line = re.sub(r'\s?([,:;\.])', r'\1', line)
      # ESGには文法的に関係のない数字が多く含まれている
      line = re.sub(r'\d{5,}', r' ', line)
      # URLの記号などを削除
      line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
      # 複数のスペースを削除
      line = re.sub('\s+', ' ', line)
      
      # Spacyを利用して段落から文章へ
      for part in list(nlp(line).sents):
        sentences.append(str(part).strip())

  return sentences

@pandas_udf('array<string>', PandasUDFType.SCALAR_ITER)
def extract_statements_udf(content_series_iter):
  """
  as loading a spacy model takes time, we certainly do not want to load model for each record to process
  we load model only once and apply it to each batch of content this executor is responsible for
  """
  
  # Spacyモデルのロード
  spacy.cli.download("en_core_web_sm")
  nlp = spacy.load("en_core_web_sm", disable=['ner'])
  
  # PDFコンテンツに対して、クレンジング、形態素解析を実施 
  for content_series in content_series_iter:
    yield content_series.map(lambda x: extract_statements(nlp, x))

# *****************************
# トランスフォーメーションを実行
# *****************************

esg_statements = esg_articles \
  .withColumn('statements', extract_statements_udf(F.col('content'))) \
  .withColumn('statement', F.explode(F.col('statements'))) \
  .filter(F.length(F.col('statement')) > 100) \
  .select('company', 'statement') \
  .cache()

esg_statements.count()
display(esg_statements)

# COMMAND ----------

# DBTITLE 1,Tokenize statements into sentences
def tokenize(sentence):
  gen = gensim.utils.simple_preprocess(sentence, deacc=True)
  return ' '.join(gen)

def lemmatize(nlp, text):
  
  # spacyを利用して文字をパース
  doc = nlp(text) 
  
  # 書く言葉を簡易的なフォームへ変換 (singular, present form, etc.)
  lemma = []
  for token in doc:
      if (token.lemma_ not in ['-PRON-']):
          lemma.append(token.lemma_)
          
  return tokenize(' '.join(lemma))

@pandas_udf('string', PandasUDFType.SCALAR_ITER)
def lemma(content_series_iter):
  """
  as loading a spacy model takes time, we certainly do not want to load model for each record to process
  we load model only once and apply it to each batch of content this executor is responsible for
  """

  # spacyモデルをロード
  spacy.cli.download("en_core_web_sm")
  nlp = spacy.load("en_core_web_sm", disable=['ner'])
  
  # テキストコンテンツを文にレマタイズする
  for content_series in content_series_iter:
    yield content_series.map(lambda x: lemmatize(nlp, x))
    
# *****************************
# 変換を適用する
# *****************************

esg_lemma = esg_statements \
  .withColumn('lemma', lemma(F.col('statement'))) \
  .select('company', 'statement', 'lemma')

display(esg_lemma)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2:  トピック モデリング
# MAGIC クレンジング後のデータセットには、よく定義されたESGステートメントのみが含まれているため、メモリに簡単に収まります（私たちのコーパスは約7,200サイズです）。このデータセットをpandas dataframeに変換し、pythonのネイティブ機能を使って視覚化します。最初のアプローチは、常にNグラム分析のための用語頻度（および逆文書頻度 - [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)を参照）を見ることです。ドメイン特有のストップワード（bank, plc, inc.など）を加えることで、単純なグラフやワードクラウドを使って関連する単語を簡単に見ることができます。

# COMMAND ----------

# DBTITLE 1,データセットをPandasに変換
esg = esg_lemma.select("company", "statement", "lemma").toPandas()

# COMMAND ----------

# DBTITLE 1,特定のストップワードを設定
# トピックモデリングに含めてはいけない文脈依存のキーワード
fsi_stop_words = [
  'plc', 'group', 'target',
  'track', 'capital', 'holding',
  'report', 'annualreport',
  'esg', 'bank', 'report',
  'annualreport', 'long', 'make'
]

# ストップワードとして会社名を追加
for fsi in [row[0] for row in esg_urls_rows]:
    for t in fsi.split(' '):
        fsi_stop_words.append(t)

# 私たちのリストは、すべての英語のストップワード+企業名+特定のキーワードを含んでいます
stop_words = text.ENGLISH_STOP_WORDS.union(fsi_stop_words)

# COMMAND ----------

# DBTITLE 1,コーパスの幅広い用語の頻度
# 7200件のレコードを1つの大きな文字列に集約して、単語の頻度でワードクラウドを実行する。
# sparkフレームワークを利用してTF分析を行い、代わりにwordcloud.generate_from_frequenciesを呼び出すことができる。
large_string = ' '.join(esg.lemma)

# サードパーティのライブラリを使用して用語の頻度を計算し、ストップワードを適用する
word_cloud = WordCloud(
    background_color="white",
    max_words=5000, 
    width=900, 
    height=700, 
    stopwords=stop_words, 
    contour_width=3, 
    contour_color='steelblue'
)

# 全てのレコードのワードクラウドを表示する
plt.figure(figsize=(10,10))
word_cloud.generate(large_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# DBTITLE 1,ビグラム解析
# バイグラムのTF-IDFフリークワンシーを計算する
bigram_tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,2), min_df=10, use_idf=True)
bigram_tf_idf = bigram_tf_idf_vectorizer.fit_transform(esg.lemma)

# バイグラムの抽出 名前
words = bigram_tf_idf_vectorizer.get_feature_names()

# トップ10のNグラムを抽出
total_counts = np.zeros(len(words))
for t in bigram_tf_idf:
    total_counts += t.toarray()[0]

count_dict = (zip(words, total_counts))
count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
words = [w[0] for w in count_dict]
counts = [w[1] for w in count_dict]
x_pos = np.arange(len(words)) 

# 上位10個のngramをプロット
plt.figure(figsize=(15, 5))
plt.subplot(title='10 most common bi-grams')
sns.barplot(x_pos, counts, palette='Blues_r')
plt.xticks(x_pos, words, rotation=90) 
plt.xlabel('bi-grams')
plt.ylabel('tfidf')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC TF-IDF を用いると、ほとんどの組織が「気候変動(climate change)」に強く焦点を当てていることがわかります。興味深いことに、ESGはリスク管理と密接に関係しており、抽出された上位4つのビグラムにも含まれています。我々は、TF-IDFとTFを用いて、[non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)または[latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)のいずれかを用いた複数のアプローチを評価しました。実験の結果、9つのトピックが私たちのコーパス文書に対して記述的であることがわかりました。`LatentDirichletAllocation`の既製のsklearnバージョンを使用して、ESGドキュメントからトピックを学習しました。

# COMMAND ----------

# DBTITLE 1,トピック モデリング
# NMFと比較して、LDAは用語頻度の確率分布を学習するため、逆文書頻度を必要としない。
word_tf_vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,1))
word_tf = word_tf_vectorizer.fit_transform(esg.lemma)

# ストップワードのシリアライズ
json_data = json.dumps([a for a in stop_words], indent=2)
f = open("/tmp/stopwords.json", "w")
f.write(json_data)
f.close()
  
# MLflowでの実験追跡
with mlflow.start_run(run_name='topic_modelling'):
  
  # 9つのトピックでLDAモデルを学習
  lda = LDA(random_state = 42, n_components = 9, learning_decay = .3)
  lda.fit(word_tf)
  
  # モデルのロギング 
  mlflow.sklearn.log_model(lda, "model")
  mlflow.log_param('n_components', '9')
  mlflow.log_param('learning_decay', '.3')
  mlflow.log_metric('perplexity', lda.perplexity(word_tf))
  mlflow.log_artifact("/tmp/stopwords.json")
  
  # 後でトピック名を添付するためにランIDを取得する
  lda_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# DBTITLE 1,トピック説明
# 9つのトピックを説明する上位N個の単語を取得する
def top_words(model, feature_names, n_top_words):
  rows = []
  for topic_idx, topic in enumerate(model.components_):
    message = ", ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    rows.append(["Topic #%d: " % (topic_idx + 1), message])
  return pd.DataFrame(rows, columns=['topic', 'keywords'])

# 検査のためにトピックのキーワードを表示し、トピックをラベル付け
tf_feature_names = word_tf_vectorizer.get_feature_names()
display(top_words(lda, tf_feature_names, 15))

# COMMAND ----------

# MAGIC %md
# MAGIC LDAモデルから抽出したトピックの説明をもとに、以下のように手動で名前を付けます。これは、デモの残りの部分で、ESGステートメントに基づいて組織を比較する方法として使用されます。

# COMMAND ----------

# DBTITLE 1,トピック名の定義
# 私たちは、以下のテーマに沿ってトピックを推定しました。
topic_names = [
  'value employees',
  'strong governance', 
  'company transformation',
  'ethical investments',
  'sustainable finance',
  'support community',
  'focus customer',
  'code of conduct',
  'green energy'
]

# トピック名のシリアライズ
json_data = json.dumps(topic_names, indent=2)
f = open("/tmp/topics.json", "w")
f.write(json_data)
f.close()

# mlflowのLDAモデルにトピック名を付ける
# run_idを再オープンして開始/終了時刻を変更することはしたくない。
# 代わりに、既存のランにアーティファクトを記録する
client = mlflow.tracking.MlflowClient()
client.log_artifact(lda_run_id, "/tmp/topics.json")

# COMMAND ----------

# DBTITLE 1,トピックを表示
# ♪シンプルなワードクラウドによる視覚化で、トピックの関連性を確保する
def word_cloud(model, tf_feature_names, index):
    
    imp_words_topic=""
    comp = model.components_[index]
    vocab_comp = zip(tf_feature_names, comp)
    sorted_words = sorted(vocab_comp, key = lambda x:x[1], reverse=True)[:50]
    
    for word in sorted_words:
        imp_words_topic = imp_words_topic + " " + word[0]
    
    return WordCloud(
        background_color="white",
        width=600, 
        height=600, 
        contour_width=3, 
        contour_color='steelblue'
    ).generate(imp_words_topic)
    
topics = len(lda.components_)
fig = plt.figure(figsize=(20, 20 * topics / 3))

# 抽出された各トピックのワードクラウドを表示
for i, topic in enumerate(lda.components_):
    ax = fig.add_subplot(topics, 3, i + 1)
    ax.set_title(topic_names[i], fontsize=20)
    wordcloud = word_cloud(lda, tf_feature_names, i)
    ax.imshow(wordcloud)
    ax.axis('off')

# COMMAND ----------

# DBTITLE 1,各ESGステートメントにトピック分布を添付
# オリジナルのデータセットにスコアを付けて、各ESGステートメントにトピック分布を添付する
transformed = lda.transform(word_tf)

# ディストリビューションから主要なトピックを見つける...
a = [topic_names[np.argmax(distribution)] for distribution in transformed]

# ... 確率的には
b = [np.max(distribution) for distribution in transformed]

# LDA出力を便利なデータフレームに集約する 
df1 = esg[['company', 'lemma', 'statement']]
df2 = pd.DataFrame(zip(a,b,transformed), columns=['topic', 'probability', 'probabilities'])
esg_group = pd.concat([df1, df2], axis=1)

# データフレームの表示
display(esg_group[['company', 'lemma', 'topic', 'probability']])

# COMMAND ----------

# DBTITLE 1,企業の中核となるESGの取り組みを比較
# 組織全体での各トピックの出現数を示す単純なピボットテーブルを作成する
esg_focus = pd.crosstab(esg_group.company, esg_group.topic)

# トピックの頻度を0から1の間で調整
scaler = MinMaxScaler(feature_range = (0, 1))

# ピボットテーブルの正規化
esg_focus_norm = pd.DataFrame(scaler.fit_transform(esg_focus), columns=esg_focus.columns)
esg_focus_norm.index = esg_focus.index

# 各FSIが学習したトピックの中で主に注力した分野を示すヒートマップをプロット
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(esg_focus_norm, annot=False, linewidths=.5, cmap='Blues')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC このマトリックスは、FSI全体のESG戦略を簡単に把握することができます。ロイヤルバンク・オブ・カナダのように従業員や民間人に重点を置く企業がある一方で、ゴールドマン・サックスやKKRのように倫理的な投資にやや重点を置く企業もあります。

# COMMAND ----------

# DBTITLE 1,トピック確率の分布を表示
# 全ての記述が明確に定義されたテーマに沿っているわけではない
# いくつかのステートメントは、より一般的で、複数のテーマにまたがっているかもしれない
esg_group.probability.hist(bins=50, figsize=(10,8), color='steelblue')

# 主要トピックの確率分布のプロット
plt.axvline(0.89, color='coral', linestyle='--')
plt.title('Primary topic distribution')
plt.xlabel('probability')
plt.ylabel('density')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC いくつかのステートメントのみしか、1つの明確なトピックに固有のものでありません。閾値0.9を適用すると、トピックに特化したステートメントが約20%あります。<br>
# MAGIC このしきい値を適用して、 **"銀行XXXは、地域社会を支援するためにどのような行動をとっているのか？"** のような質問に答えるトピック固有のステートメントを抽出します。

# COMMAND ----------

# DBTITLE 1,各トピックのキー・ステートメントを取得
# 与えられたトピックに関連するステートメントを抽出
topic_discussions = esg_group[esg_group['topic'] == 'value employees']

# 確率分布で指定されているように、一般的な議論ではなく、特定のトピックのみを求めている
topic_discussions = topic_discussions[topic_discussions['probability'] > 0.89]

# より特定的なトピックへのアクセスを優先
topic_discussions = topic_discussions.sort_values('probability', ascending=False)

rows = [] 
for i, row in topic_discussions.iterrows():
  rows.append([row.company, row.probability, row.statement])

# 選択されたトピックのステートメントのデータフレームを表示する
display(pd.DataFrame(rows, columns=['company', 'probability', 'statement']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3:  ESGの主な取り組み
# MAGIC 前のステップでは、7200の大規模コーパス文書からトピックを抽出するために、Latent Dirichlet allocation（潜在的ディリクレ配分）を使用しました。これにより、定義された9つのトピックのそれぞれをカバーするように、各ステートメントの確率分布が返されます。これは、ESGレポートをプログラム的に要約する際の大きなヒントにはなりますが、ある組織の声明が他の組織と比較してどのような特異性を持っているかについては、あまり言及していません。言い換えれば、議論されているのが一般的な用語なのか、企業固有のESG戦略なのかを区別することはできません。我々の目的は、LDAから得られた確率分布を、上位のクラスタリングアルゴリズム（この場合は[KMeans](https://en.wikipedia.org/wiki/K-means_clustering)）の入力ベクトルとして使用することです。ステートメントを類似したバケットにグループ化することで、ESGステートメントが標準からどのように逸脱しているかに関して、その関連性にアクセスできるようになります。このようなアプローチは、異常検知と同様に、各組織にとって重要な戦略的イニシアチブをもたらします。

# COMMAND ----------

# DBTITLE 1,理想的なクラスター数について
# KMeansの入力ベクトルとして確率分布を抽出
X_train = list(esg_group.probabilities)

# それでも、単純な "エルボー法 "を用いて、クラスタリングの妥当性を確認
# kの値を変えて、各点から最も近い中心までの二乗距離の総和を求める
wsses = []
for k in [5, 8, 10, 20, 30, 50, 80, 100]:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(X_train)
  wsse = np.sum([np.min(x)**2 for x in kmeans.transform(X_train)]) 
  wsses.append([k, wsse])
  
# WSSEとKを単純にプロットして最適なK値を見つける
wsse_df = pd.DataFrame(wsses, columns=['k', 'wsse'])
display(wsse_df)

# COMMAND ----------

# MAGIC %md
# MAGIC この単純な方法により、類似したステートメントをグループ化するための理想的なクラスター数は、15～20程度であることがわかります。

# COMMAND ----------

# DBTITLE 1,ステートメントをバケットにグループ化し、最も近いセンターからの距離を得る
# MLflowでの実験トラック
with mlflow.start_run(run_name='clustering'):
  
  # KMeansモデルを、適切なKの値でトレーニングします。
  kmeans = KMeans(n_clusters=20, random_state=42)
  kmeans.fit(X_train)
  
  # モデルのロギング 
  mlflow.sklearn.log_model(kmeans, "model")
  mlflow.log_param('n_clusters', '15')
  mlflow.log_metric('wsse', np.sum([np.min(x)**2 for x in kmeans.transform(X_train)]))
  
  # 実験IDの取得
  cluster_run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# DBTITLE 1,最も近いクラスターにステートメントを割り当てる
# 各点の最も近いクラスタへの最小距離を見つける
y_dist = [np.min(x) for x in kmeans.transform(X_train)]
dist_df = pd.DataFrame(zip(y_dist), columns=['distance'])
esg_group_dist = pd.concat([esg_group, dist_df], axis=1)

# COMMAND ----------

# DBTITLE 1,抽出されたイニシアチブをデータとして永続化 
# 競合他社と比較したトピックの関連性としてクラスター距離が与えられた場合, 我々は
# その決定をエンドユーザーに委ね, SQL機能を使って業界一般と企業固有の閾値を適用する.
spark.createDataFrame(esg_group_dist) \
  .write \
  .format("delta") \
  .saveAsTable("esg.reports")

# COMMAND ----------

# DBTITLE 1,特定のESG戦略を有する企業を探す
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   t.company,
# MAGIC   t.topic,
# MAGIC   t.statement
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     e.company,
# MAGIC     e.topic,
# MAGIC     e.probability,
# MAGIC     e.distance,
# MAGIC     LOWER(e.statement) AS statement,
# MAGIC     dense_rank() OVER (PARTITION BY e.company, e.topic ORDER BY e.distance DESC) as rank
# MAGIC   FROM esg.reports e
# MAGIC ) t
# MAGIC WHERE t.rank = 1
# MAGIC AND t.topic IN ('green energy')
# MAGIC ORDER BY company, topic, rank

# COMMAND ----------

# DBTITLE 1,ゴールドマン・サックスの重要な戦略的イニシアチブを示す
# MAGIC %sql
# MAGIC 
# MAGIC WITH ranked (
# MAGIC   SELECT 
# MAGIC     e.topic, 
# MAGIC     e.statement, 
# MAGIC     e.company,
# MAGIC     dense_rank() OVER (PARTITION BY e.company, e.topic ORDER BY e.probability DESC) as rank
# MAGIC   FROM esg.reports e
# MAGIC )
# MAGIC 
# MAGIC SELECT 
# MAGIC   t.topic,
# MAGIC   t.statement
# MAGIC FROM ranked t
# MAGIC WHERE t.company = 'goldman sachs' 
# MAGIC AND t.rank = 1

# COMMAND ----------

# MAGIC %md
# MAGIC 複雑なPDF文書をNLP技術を用いて要約し、重要なイニシアティブを抽出することで、これらの洞察をレポートの形でアセットマネージャーやリスクオフィサーに提供することができます。
