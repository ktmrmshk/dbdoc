# Databricks notebook source
# MAGIC %md 
# MAGIC # MLflowとDelta Lakeを用いた機械学習モデルのトラッキング
# MAGIC 
# MAGIC よくある話ですが、データチームがモデルをトレーニングし、本番環境にデプロイし、一時的にはすべてが順調に進みます。しかし、モデルがおかしな予測をするようになり、すぐにモデルの検査とデバッグが必要になります。
# MAGIC 
# MAGIC このノートブックでは、[MLflow](http://mlflow.org)と[Delta Lake](http://delta.io)を使って、モデルのトレーニングランを簡単に追跡(トラッキング)、可視化、再現、デバッグを容易にする方法を紹介します。具体的には以下の内容になります。
# MAGIC 
# MAGIC 1. MLパイプラインを構築で使用したデータの特定日時のスナップショットを追跡し、再現する。
# MAGIC 2. 特定のスナップショットデータでトレーニングされたモデルを識別する。
# MAGIC 3. 過去のスナップショットデータでトレーニングを再実行する（例：古いモデルを再現する）。
# MAGIC 
# MAGIC このノートブックでは、Delta Lakeを使用してデータのバージョン管理と「タイムトラベル」機能（古いバージョンのデータを復元する機能）を提供し、MLflowを使用してデータを追跡し、過去のトレーニング結果を検索・参照するケースを見ていきます。
# MAGIC 
# MAGIC **クラスタの要件**:
# MAGIC * Databricks Runtime 7.0 MLもしくはそれ以上
# MAGIC * Maven library `org.mlflow:mlflow-spark:1.11.0`がインストールされている

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 問題設定: 貸し手にとっての「不良債権」を分類する
# MAGIC 
# MAGIC このノートでは、Lending Clubデータセットの分類問題に取り組み、クレジットスコア、クレジットヒストリー、その他の特徴の組み合わせに基づいて、「不良債権」（採算が取れない可能性の高いローン）を特定することを目的としています。
# MAGIC 
# MAGIC 最終的なゴールは、融資担当者が融資を承認するかどうかを決定する際に使用できる、解釈可能なモデルを作成することです。このようなモデルは、貸し手にとっては有益な情報を提供し、かつ、借り手にとっても即座に見積もりや審査の結果がすぐにわかるようになります。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### データ
# MAGIC 
# MAGIC 使用するデータはLending Clubの公開データです。2012年から2017年までのすべての資金提供されたローンが含まれています。各ローンには、申請者が提供した申請者情報のほか、現在のローンステータス（Current、Late、Fully Paidなど）や最新の支払い情報が含まれています。データの全体像は、[data dictionary](https://resources.lendingclub.com/LCDataDictionary.xlsx)をご覧ください。
# MAGIC 
# MAGIC ![Loan_Data](https://preview.ibb.co/d3tQ4R/Screen_Shot_2018_02_02_at_11_21_51_PM.png)
# MAGIC 
# MAGIC 
# MAGIC https://www.kaggle.com/wendykan/lending-club-loan-data

# COMMAND ----------

# MAGIC %md 
# MAGIC ### セットアップ: ストレージ(DBFS)上にDeltaテーブルを作成する
# MAGIC 
# MAGIC DBFSに格納されている既存のParquetテーブルを変換して、Delta Lake形式のサンプルデータを生成します。

# COMMAND ----------

from pyspark.sql.functions import *

# もし既にテーブルが存在していたら削除する
DELTA_TABLE_DEFAULT_PATH = "/ml/loan_stats.delta"
dbutils.fs.rm(DELTA_TABLE_DEFAULT_PATH, recurse=True)

# Lending Clubのデータを読み込む(既にDatabricksが用意しているサンプルデータ)
lspq_path = "/databricks-datasets/samples/lending_club/parquet/"
data = spark.read.parquet(lspq_path)

# 前処理: 必要なカラムだけ抜き出す
features = ["loan_amnt",  "annual_inc", "dti", "delinq_2yrs","total_acc", "total_pymnt", "issue_d", "earliest_cr_line"]
raw_label = "loan_status"
loan_stats_ce = data.select(*(features + [raw_label]))

print("------------------------------------------------------------------------------------------------")
print("不良債権のラベルを作成。このラベルには、ローンのチャージオフ、デフォルト、返済遅延などが含まれる...")
loan_stats_ce = (
  loan_stats_ce
  .filter(loan_stats_ce.loan_status.isin(["Default", "Charged Off", "Fully Paid"]))
  .withColumn("bad_loan", ( ~(loan_stats_ce.loan_status == "Fully Paid")).cast("string") )
)

# 100000レコードにする(Community Editionでも実行できるように)
loan_stats_ce = loan_stats_ce.orderBy(rand()).limit(10000) 


print("------------------------------------------------------------------------------------------------")
print("数値列を適切な型にキャストする...")
loan_stats_ce = (
  loan_stats_ce
  .withColumn('issue_year',  substring(loan_stats_ce.issue_d, 5, 4).cast('double'))
  .withColumn('earliest_year', substring(loan_stats_ce.earliest_cr_line, 5, 4).cast('double'))
  .withColumn('total_pymnt', loan_stats_ce.total_pymnt.cast('double'))
)

loan_stats_ce = (
  loan_stats_ce
  .withColumn('credit_length_in_years', (loan_stats_ce.issue_year - loan_stats_ce.earliest_year))   
)

# 最後にDeltaフォーマットでストレージに書き出す
loan_stats_ce.write.format("delta").mode("overwrite").save(DELTA_TABLE_DEFAULT_PATH)

# 上記のdataframeの中身を見てみる。
display(loan_stats_ce)

# COMMAND ----------

# MAGIC %md ## 1. 再現性のためにデータのバージョンとパスをトラックする
# MAGIC 
# MAGIC 
# MAGIC このノートブックは、データバージョンとデータパスをウィジェットで入力パラメータとして受け付けるので、将来、明示的に指定されたデータバージョンとパスに対してノートブックの実行を再現することができます。データのバージョン管理はDelta Lakeを使用する利点であり、前のバージョンのデータセットを保存し、後で復元することができます。

# COMMAND ----------

# データのパスとversionをノートブックの上部にあるウィジットUIから引いてくる
dbutils.widgets.text(name="deltaVersion", defaultValue="1", label="Table version, default=latest")
dbutils.widgets.text(name="deltaPath", defaultValue="", label="Table path")

data_version = None if dbutils.widgets.get("deltaVersion") == "" else int(dbutils.widgets.get("deltaVersion"))
DELTA_TABLE_DEFAULT_PATH = "/ml/loan_stats.delta"
data_path = DELTA_TABLE_DEFAULT_PATH if dbutils.widgets.get("deltaPath")  == "" else dbutils.widgets.get("deltaPath")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deltaテーブルからデータを読み込む
# MAGIC 
# MAGIC ウィジェットで指定されたデータパスとバージョンを使用して、Delta Lakeフォーマットでデータをロードバックします。

# COMMAND ----------

# デフォルトで最新のデータを読み込む。指定があれば、そのバージョンを読み込む。
if data_version is None:
  from delta.tables import DeltaTable  
  delta_table = DeltaTable.forPath(spark, data_path)
  version_to_load = delta_table.history(1).select("version").collect()[0].version  
else:
  version_to_load = data_version

loan_stats = (
  spark
  .read
  .format("delta")
  .option("versionAsOf", version_to_load) # <= Version指定のオプション!
  .load(data_path)
)

# データを確認
display(loan_stats)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deltaテーブルのヒストリを確認
# MAGIC 
# MAGIC Deltaテーブルのすべてのトランザクションは、挿入、更新、削除、マージ、挿入の初期セットを含めて、テーブル内に保存されます。

# COMMAND ----------

# Deltaファイルとテーブルデータ(Hiveメタストア)を関連づける
spark.sql("DROP TABLE IF EXISTS loan_stats")
spark.sql("CREATE TABLE loan_stats USING DELTA LOCATION '" + DELTA_TABLE_DEFAULT_PATH + "'")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- テーブルのヒストリを確認する
# MAGIC DESCRIBE HISTORY loan_stats

# COMMAND ----------

# MAGIC %md 
# MAGIC ### モデルのトレーニング (クロスバリデーション + パイパーパラメータチューニング)
# MAGIC 
# MAGIC 
# MAGIC Spark MLlibを使ってMLパイプラインをチューニングします。チューニングを行った際のメトリクスやパラメターは自動的にMLflowにトラッキングされ、後で確認することができます。

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import mlflow.spark
from pyspark.sql import SparkSession

# MLflowの自動ログトラッキング機能を有効にする
mlflow.spark.autolog()

# ヘルパー関数: 2値分類モデル(Binary Classification)をクロスバリデーションで学習する際に使用する
def _fit_crossvalidator(train, features, target):
  """
  Helper function that fits a CrossValidator model to predict a binary label
  `target` on the passed-in training DataFrame using the columns in `features`
  :param: train: Spark DataFrame containing training data
  :param: features: List of strings containing column names to use as features from `train`
  :param: target: String name of binary target column of `train` to predict
  """
  # トレーニングデータを読み込む
  train = train.select(features + [target])
  
  # 前処理のパイプライン(欠損値補完 -> ベクトル化 -> ラベルカラムのインデックス化)
  model_matrix_stages = [
    Imputer(inputCols = features, outputCols = features),
    VectorAssembler(inputCols=features, outputCol="features"),
    StringIndexer(inputCol="bad_loan", outputCol="label")
  ]
  
  # 学習モデル(Logistic回帰)のインスタンス化
  lr = LogisticRegression(maxIter=10, elasticNetParam=0.5, featuresCol = "features")
  
  # パイプラインを構成
  pipeline = Pipeline(stages=model_matrix_stages + [lr])
  
  # Gridサーチのハイパーパラメータの範囲指定
  paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
  
  # クロスバリデーション学習器のインスタンス化
  crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=BinaryClassificationEvaluator(),
                            numFolds=5)
  # 学習実行
  cvModel = crossval.fit(train)
  
  # 最良のモデルを返す
  return cvModel.bestModel

# COMMAND ----------

# 実際に学習を実施する(上記で定義したヘルパー関数を呼び出す)
features = ["loan_amnt",  "annual_inc", "dti", "delinq_2yrs","total_acc", "credit_length_in_years"]
glm_model = _fit_crossvalidator(loan_stats, features, target="bad_loan")

# 学習結果のROCを出力 (モデルの性能が悪い!)
lr_summary = glm_model.stages[len(glm_model.stages)-1].summary
display(lr_summary.roc)

# COMMAND ----------

# Accuracyを確認 (モデルの性能が悪い!)
print("ML Pipeline accuracy: %s" % lr_summary.accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### NotebookのサイドバーにからMLflowでトラックした学習結果(Experiment/Runs)をかくする
# MAGIC 
# MAGIC 上記のモデル学習コードでは、MLflow実行時に自動的にメトリクスやパラメータが記録され、[MLflow Runs Sidebar](https://databricks.com/blog/2019/04/30/introducing-mlflow-run-sidebar-in-databricks-notebooks.html)で確認することができます。右上の「Experiment」をクリックすると、「Experiment Runs」サイドバーが表示されます。

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 特徴量エンジニアリング: データスキーマを発展させる
# MAGIC 
# MAGIC Delta Lakeを使うとデータセットの過去のバージョンを容易にロールバックでき、モデルのパフォーマンスを向上させる可能性のある機能エンジニアリングを行うことができます。まず、ローンごとに利潤・損失の合計額を追跡する機能を追加します。

# COMMAND ----------

print("------------------------------------------------------------------------------------------------")
print("1回の融資で得られる金額と失われる金額の合計を算出する...")
loan_stats_new = (
  loan_stats
  .withColumn('net', round( loan_stats.total_pymnt - loan_stats.loan_amnt, 2))
)


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 更新されたテーブルは元のテーブルとスキーマが異なるので、それを明示的にオプション指定(`mergeSchema`オプション)して上書きする。安全にスキーマを発展(変更)させることができる。

# COMMAND ----------

(
  loan_stats_new
  .write
  .option("mergeSchema", "true") # <= スキーマ進化!
  .format("delta")
  .mode("overwrite")
  .save(DELTA_TABLE_DEFAULT_PATH)
)

# COMMAND ----------

# オリジナルのデータと、今回の変更したデータのスキーマ差分を確認する
set(loan_stats_new.schema.fields) - set(loan_stats.schema.fields)

# COMMAND ----------

# 変更後のデータの確認
display( spark.sql('SELECT * FROM loan_stats') )

# COMMAND ----------

# MAGIC %md 
# MAGIC 更新されたデータでモデルを再学習し、その性能を元のデータと比較する。

# COMMAND ----------

# 更新したデータで再度学習する
# (つまり、先ほど定義したヘルパー関数(クロスバリデーション学習)に更新データを入力)
glm_model_new = _fit_crossvalidator(loan_stats_new, features + ["net"], target="bad_loan")

# 再度、ROCで評価　(性能が改善できた)
lr_summary_new = glm_model_new.stages[len(glm_model_new.stages)-1].summary
display(lr_summary_new.roc)

# COMMAND ----------

# accuracryを確認　(性能が改善できた)
print("ML Pipeline accuracy: %s" % lr_summary_new.accuracy)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. オリジナルデータで学習した結果(run)を検索・参照する
# MAGIC 
# MAGIC モデルの精度は、特徴量エンジニアリングによって、80%から95%に向上しました。では、元のデータセットで構築したすべてのモデルを、特徴量エンジニアリングを行ったデータセットで再学習したらどうなるのでしょうか？モデルの性能は同様に向上するのでしょうか？
# MAGIC 
# MAGIC 元のデータセットに対して実施された他のランを特定するには、MLflowの`mlflow.search_runs`というAPIを使います。

# COMMAND ----------

mlflow.search_runs(filter_string="tags.sparkDatasourceInfo LIKE 'path=%{path},version={version},%'".format(path=data_path, version=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. データのスナップショットでロードバックを実施し学習結果を再現する
# MAGIC 
# MAGIC 最後に、モデルの再トレーニングに使用するために、データの特定のバージョンをロードバックすることができます。これを行うには、ノートブック上部のウィジェットを`Table version`=`1`（特徴量エンジニアリング後のデータに対応）に更新し、このノートブックのセクション1から再実行します。

# COMMAND ----------


