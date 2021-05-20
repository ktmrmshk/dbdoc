# Databricks notebook source
# MAGIC %md 
# MAGIC # 風車・ウィンドタービンの予知保全
# MAGIC 
# MAGIC このデモでは、風車のウィンドタービンの予知保全を例にとり、以下のDatabricksの特徴を説明いたします。
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC * **DatabricksのWorkspaceとクラスタ管理** (Workspace: すぐに作業スタート)
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC * **Databricks上での機械学習とモデルの管理** (Spark MLlib, MLflow: シンプル・首尾一貫な機械学習の運用)
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/spark.png" width=120 />
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/mlflow_logo.ong.png" width=150 />
# MAGIC * **バッチ処理・ストリーミング処理の統合・学習モデル適用・推定** (Delta Lake: シームレスな本番適用)
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/deltalake.png" width=100 />

# COMMAND ----------

# DBTITLE 1,デモのシナリオ
# MAGIC %md
# MAGIC 
# MAGIC * **目的**: 風車のウィンドタービンの故障をセンサーデータを基に検出する
# MAGIC * **データ**: タービンの周辺部に設置された8種類のセンサー(振動センサーなど)
# MAGIC   - 既にウィンドタービンが「正常である」「ダメージを受けている」のタグ付けされたデータは準備済み
# MAGIC * **アプローチ**: いくつかのアルゴリズムで学習し、精度が高いモデルを構築する
# MAGIC * **本番展開**: 現場から逐次送信されてくるデータ(ストリーミングデータ)に学習したモデルを適用し、故障検出をリアルタイムに実施する

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC このデモではダメージを受けたウィンドタービンを見つけるための異常検知を見ていきます。実際に、1つの故障したウィンドタービンがもたらす損失は1日あたり数十万円にのぼると言われいます。
# MAGIC 
# MAGIC 使用するデータはウィンドタービンのギアボックスに設置されたバイブレーションセンサーから送出されたものになります(下図参照)。このデモでは勾配ブースティング決定木(Gradient Boosted Tree)を使用してどのデータが故障を予知するのかを推定していきます。
# MAGIC 
# MAGIC *センサーの設置部位と収集データは下記の通りです*
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width=800 />
# MAGIC 
# MAGIC ![Location](https://s3-us-west-2.amazonaws.com/databricks-demo-images/wind_turbine/wtsmall.png)
# MAGIC 
# MAGIC https://www.nrel.gov/docs/fy12osti/54530.pdf
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC Gearbox Reliability Collaborative
# MAGIC 
# MAGIC https://openei.org/datasets/dataset/wind-turbine-gearbox-condition-monitoring-vibration-analysis-benchmarking-datasets
# MAGIC 
# MAGIC https://www.nrel.gov/docs/fy12osti/54530.pdf

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC **機械学習のワークフロー**
# MAGIC 
# MAGIC 1. 生データの収集
# MAGIC 1. 前処理・特徴量エンジニアリング
# MAGIC 1. モデルの学習
# MAGIC 1. チューニング
# MAGIC 1. モデルの評価
# MAGIC 
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/ML-workflow.png" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Workspaceとクラスタの管理
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/Databricks_Workspace.png" width="1200"/>
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC <h3>Databricksを使用することで、クラスタ管理が自動化されオペレーションの時間的・人的・金銭的コストが大幅に削減できます。</h3>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. データの整形(ETL)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/ML-workflow-ETL.png" width="800"/>

# COMMAND ----------

from pyspark.ml import *
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Schemaの設定
#   => 全てDouble型のデータ
schema = StructType(
  [StructField(col, DoubleType(), False) for col in ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE"]]
)

# データの読み込み (ダメージありタービンのデータ)
# 今回、データはS3上にCSVで保存されているものを使用する
turbine_damaged = (
  spark.read
  .schema(schema)
  .csv("/mnt/databricks-datasets-private/ML/wind_turbine/D*")
  .drop("TORQUE")
  .drop("SPEED")
  .repartition(150)
  .cache()
)

# データの読み込み (正常タービンのデータ)
turbine_healthy = ( 
  spark.read
  .schema(schema)
  .csv("/mnt/databricks-datasets-private/ML/wind_turbine/H*")
  .drop("TORQUE")
  .drop("SPEED")
  .repartition(150)
  .cache()
)
  
# SQLを実行するためにDataFrameからViewを作成しておく
turbine_damaged.createOrReplaceTempView("turbine_damaged")
turbine_healthy.createOrReplaceTempView("turbine_healthy")

# ダメージありタービンデータと正常タービンデータを一つのDataFrameにまとめる (学習処理のため)
df = (
  turbine_healthy
  .withColumn("ReadingType", lit("HEALTHY"))
  .union(
    turbine_damaged.withColumn("ReadingType", lit("DAMAGED"))
  )
)

# ランダムに並び替える (ダメージありデータ/正常データをランダムに並べ替える)
df = df.orderBy(rand()).cache()

# df, df_rest = df.randomSplit([0.02, 0.98], seed=42)
# df = df.orderBy(rand()).cache()

# COMMAND ----------

# 読み込んだデータのレコード数を確認
print('ダメージありタービンのレコード数 =>', turbine_damaged.count())
print('正常タービンのレコード数 =>', turbine_healthy.count())

# COMMAND ----------

# ダメージありタービンのDataFrame
display(turbine_damaged)

# COMMAND ----------

# 正常タービンのDataFrame
display(turbine_healthy)

# COMMAND ----------

# ダメージありデータと正常データをまとめたDataFrame
display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 2. データ探索
# MAGIC 
# MAGIC データが手元に用意できたので、続いてデータの探索をしていきましょう。
# MAGIC 
# MAGIC ここでは各センサーの平均、最大/最小、分散などの主要値から見ていきましょう。
# MAGIC 
# MAGIC 例えば、AN9の分散がダメージあり/正常で大きく違うことがわかります。

# COMMAND ----------

# 正常タービンデータの主要値
display(turbine_healthy.describe())

# COMMAND ----------

# ダメージありタービンデータの主要値
display(turbine_damaged.describe())

# COMMAND ----------

# ヒストグラムでAN9の分布を確認してみましょう。
# ダメージあり場合、やはり分散が大きくなっていることがわかります。

display(df)

# COMMAND ----------

# AN9の値をさらに可視化して比較してみましょう。
#
# DatabricksのNotebookではテーブルデータから可視化することができます
#
# この結果からダメージありの場合は分散が大きいことがわかります。(上下に開いている)
display(df)

# COMMAND ----------

# Q-Qプロットでも同じ結果を得ます。

display(df)

# COMMAND ----------

# Boxプロットも同様です。

display(df)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 3. モデルの構築
# MAGIC 
# MAGIC ここではPySparkのML Pipelineを用いたワークフローに沿ってモデルを構築していきます。
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/ML-workflow-ML.png" width="800"/>

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

# 特徴量を示すカラムを指定
featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]

# データの前処理を定義する
stages = [
  VectorAssembler(inputCols=featureCols, outputCol="va"),  # ベクトル化 (入力する特徴量を一つのカラムにベクターとしてまとめる。MLlibライブラリを使う上で必要)
  StandardScaler(inputCol="va", outputCol="features"),     # 値の標準化
  StringIndexer(inputCol="ReadingType", outputCol="label") # カテゴリ変数のインデックス化(数値化)
]

# 上記で定義した前処理のパイプラインを作成
pipeline = Pipeline(stages=stages)

# パイプラインに整形したDataFrameを入力し、前処理済DataFrameを作成する
featurizer = pipeline.fit(df)
featurizedDf = featurizer.transform(df)

# COMMAND ----------

# 前処理済DataFrame
display(featurizedDf)

# COMMAND ----------

# トレーニングデータとテストデータに分割する(80%/20%)
train, test = featurizedDf.select(["label", "features"]).randomSplit([0.8, 0.2])
train.cache()
test.cache()

# トレーニングデータ・テストデータのレコード数
print(train.count())
print(test.count())

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 勾配ブースティング決定木のオブジェクトを作成する
#   (説明変数、目的変数のカラムを指定、かつ、最大iterationを5に設定)
gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)

# Gridサーチ (ブースト木のDepthを4,5,6に変化させて性能を評価する)
grid = ParamGridBuilder().addGrid(
    gbt.maxDepth, [4, 5, 6]
).build()

# バイナリクラスの評価器のオブジェクトを作成
ev = BinaryClassificationEvaluator()

# 3-foldでクロスバリデーションを実施する (=>  クロスバリデーション x Gridサーチ)
cv = CrossValidator(estimator=gbt, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. モデルの評価
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/ML-workflow-Evaluation.png" width="800"/>
# MAGIC 
# MAGIC モデルの評価指標はいくつかあります。ここでは、以下の3つの指標をみていきたいと思います。
# MAGIC 
# MAGIC * ROC曲線 (Area Under ROC Curve)
# MAGIC * 特徴量の重要度 (Feature Importance)
# MAGIC * 精度(Accuracy)
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/Contingency-matrix-and-measures-calculated-based-on-it-2x2-contigency-table-for.png" width="600" />
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/roc_intro3.png" width="400" />

# COMMAND ----------

# テストデータを学習したモデルに入力して、予測をする
predictions = cvModel.transform(test)

# 予測結果
# display(predictions)

# ROC曲線のAUC値(Area Under the Curve): AUROC
print('AUC値 => {}'.format( ev.evaluate(predictions )))

# COMMAND ----------

# モデルの木構造を確認する

bestModel = cvModel.bestModel
print(bestModel.toDebugString)

# COMMAND ----------

# 特徴量の重要度 (Feature Importance)を確認する
bestModel.featureImportances

# COMMAND ----------

# 上記の可視化 (Pieチャート)
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### (再掲)
# MAGIC 
# MAGIC <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width=800 />

# COMMAND ----------

# SparkではPython, Scala, R, SQLを等価に扱えます。
# ここでは、あえてSQLを使って精度(Accuracy)を計算してみましょう。
#
# SQLを使用するにはDataFrameからViewを作成します。
# 作成したViewに対してSQLを実施します。
predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql  /* <= このブロックでSQLを実行するためのマジックコマンド */
# MAGIC SELECT avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) AS accuracy
# MAGIC FROM predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 比較としてランダムフォレストで学習モデルを構築
# MAGIC 
# MAGIC 比較のため、上記の勾配ブースティング決定木に加えて、ランダムフォレスト(Random Forest)でもモデルの学習・評価を実施してみましょう。

# COMMAND ----------

# 勾配ブースティング決定木の場合と同様の設定でランダムフォレストのモデルを構築する

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

grid = ParamGridBuilder().addGrid(
    rf.maxDepth, [4, 5, 6]
).build()


# we use the featurized dataframe here; we'll use the full pipeline when we save the model
# we use estimator=rf to access the trained model to view feature importance
# stages += [rf]
# p = Pipeline(stages = stages)

# 3-fold cross validation
cv = CrossValidator(estimator=rf, \
                    estimatorParamMaps=grid, \
                    evaluator=ev, \
                    numFolds=3)

cvModel = cv.fit(train)

# COMMAND ----------

# 構築したランダムフォレストのモデルを評価する

# テストデータを学習したモデルに入力して、予測をする
predictions = cvModel.transform(test)

# 予測結果
# display(predictions)

# ROC曲線のAUC値(Area Under the Curve): AUROC
# ev.evaluate(predictions)
#predictions = cvModel.transform(test)

print('AUC値 => {}'.format( ev.evaluate(predictions )))

# COMMAND ----------

# ランダムフォレストの決定木の構造を確認する

bestModel_rf = cvModel.bestModel
print(bestModel_rf.toDebugString)

# COMMAND ----------

# ランダムフォレストの特徴量の重要度 (Feature Importance)を確認する

bestModel_rf.featureImportances

# COMMAND ----------

# 上記の可視化
weights = map(lambda w: '%.10f' % w, bestModel_rf.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")

display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

# 精度(Accuracy)をSQLで計算するためにViewを作成
predictions.createOrReplaceTempView("predictions")

# COMMAND ----------

# MAGIC %sql /* 精度(Accuracy)を計算 */
# MAGIC SELECT avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) AS accuracy 
# MAGIC FROM predictions

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 6. MLflowによるモデル管理
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/mlflow.png" width="1200px">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. 学習したモデルのデプロイメント (Streaming方式)
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/20210222_deployment_pattern.png" width="1200px">
# MAGIC 
# MAGIC <br>
# MAGIC 
# MAGIC * このデモでは **ストリーミング方式** で学習したモデルを使用する例を見て行きます。
# MAGIC * Databricksの**Delta Lake**によりストリーミングデータをバッチ処理と同等に扱えます。
# MAGIC * 風車から逐次送られてくるセンサーデータ(ストリーミングデータ)に学習したモデルを適用し、**リアルタイムでタービンの故障検出**を実施していきましょう。
# MAGIC 
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/deltalake.png" width=100 />
# MAGIC <br>
# MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta_architecture_demo.gif" width="1200px">

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

rf = RandomForestClassifier(labelCol="label", featuresCol="features")

stages += [rf]
stages

train.unpersist()
test.unpersist()
train, test = df.randomSplit([0.8, 0.2])
train.cache()
test.cache()

cv = CrossValidator(estimator=Pipeline(stages=stages), \
                    estimatorParamMaps=grid, \
                    evaluator=BinaryClassificationEvaluator(), \
                    numFolds=3)

cvModel = cv.fit(train)
cvModel.bestModel

model = cvModel.bestModel

# COMMAND ----------

model.save('/home/masahiko.kitamura@databricks.com/turbin_predictive_maintanance/rf.model')
type(model)

# COMMAND ----------

# MAGIC %fs ls /home/masahiko.kitamura@databricks.com/turbin_predictive_maintanance/rf.model

# COMMAND ----------

from pyspark.ml.pipeline import PipelineModel
model = PipelineModel.load('/home/masahiko.kitamura@databricks.com/turbin_predictive_maintanance/rf.model')
type(model)

# COMMAND ----------


# ストリミーミングデータのソース(情報源)を擬似的に作成

streamingData = df.repartition(300)
streamingData.write.mode('overwrite').format('parquet').save('/tmp/masahiko.kitamura@databricks.com/streamingdata/')

# COMMAND ----------

# センサデータのストリーミングを受信

inputStream = (
  spark 
  .readStream 
  .schema(schema)
  .option("maxFilesPerTrigger", 1)
  .parquet("/tmp/masahiko.kitamura@databricks.com/streamingdata/")
)

# COMMAND ----------

# ストリーミングで受信したセンサーデータに対して、学習モデルを適用して推定する。
scoredStream = (
  model
  .transform(inputStream)
  .createOrReplaceTempView("stream_predictions")
)

# COMMAND ----------

# MAGIC %sql /* ストリーミングデータの推定結果の可視化 */
# MAGIC SELECT prediction, count(1) AS count 
# MAGIC FROM stream_predictions 
# MAGIC GROUP BY prediction

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #まとめ
# MAGIC 
# MAGIC このデモでは、風車のウィンドタービンの予知保全を例にとり以下のDatabricksの特徴を説明いたしました。
# MAGIC 
# MAGIC * **DatabricksのWorkspaceとクラスタ管理**
# MAGIC   - => ブラウザを開いて、すぐに作業をスタート。すぐに使える機械学習ライブラリ。 <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC 
# MAGIC * **Databricks上での機械学習とモデルの管理**
# MAGIC   - => 一つのプラトフォーム上でシンプルかつ首尾一貫した機械学習を実現。MLflowによる学習モデルの管理。<img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/spark.png" width=120 />
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/mlflow_logo.ong.png" width=150 />
# MAGIC 
# MAGIC * **バッチ処理・ストリーミング処理の統合・学習モデル適用・推定**
# MAGIC   - => バッチ・ストリーミングを統一的に扱えるDelta Lake。機械学習をシームレスに本番適用。　<img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/deltalake.png" width=100 />
