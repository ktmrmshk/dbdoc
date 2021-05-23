# Databricks notebook source
# MAGIC %md
# MAGIC # モデルドリフトの検知と対応
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作成者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>作成日</td><td>2021/04/15</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>DBR</td><td>8.1 ML</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルドリフトの概要
# MAGIC 時間の経過と共に、モデルの予測精度が悪化していく現象。

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルドリフトの種類
# MAGIC * **コンセプトドリフト**
# MAGIC   * 目的変数の統計的な性質が変化すること
# MAGIC   * 例: 今までに無かった新しい詐欺手法の発生
# MAGIC * **データドリフト**
# MAGIC   * 入力データの統計的な性質が変化すること
# MAGIC   * 例: データの季節性、個人の好みの変化
# MAGIC * **上流のデータの変更**
# MAGIC   * データパイプラインの上流の処理や運用の変更
# MAGIC   * 例: 単位の変化（華氏 → 摂氏、インチ → センチメートル）、センサーデータの故障によるデータの欠損や不正な値の送信

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルドリフトへの対応方法
# MAGIC 継続的に予測の質を監視し、変化があった際には、モデルの再訓練を実施する。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/09/model_drift.jpg" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks でのモデルドリフトの検知と対応

# COMMAND ----------

# MAGIC %md
# MAGIC * **モデルの訓練**
# MAGIC   * Delta Lake で入力データを保管。
# MAGIC   * Databricks Runtime for ML でモデルを訓練。
# MAGIC   * MLflow で、全ての実験をトラッキングし、また、全てのモデルの全てのバージョンを管理。
# MAGIC * **デプロイメント**
# MAGIC   * MLflow からモデルをロードし、推論を実施。
# MAGIC   * 推論結果を Delta Lake に記録。
# MAGIC * **モデルドリフトの検知と対応**
# MAGIC   * 実際の値と、Delta Lake に記載された推論結果をもとに予測精度を算出。
# MAGIC   * 予測精度が悪化していないか（モデルドリフトが発生していないか）を監視。
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/09/model_drift_evaluation_workflow.png" />

# COMMAND ----------

# MAGIC %md 
# MAGIC ## デモ: ガラス製品の品質管理

# COMMAND ----------

# MAGIC %md
# MAGIC ### 目的
# MAGIC * ガラス製品の品質を予測し、等級付けをすること。
# MAGIC 
# MAGIC ### 使用するデータ
# MAGIC * sensor_reading: 製造装置からのセンサーデータ。各製品について、製造時の温度、圧力、処理時間が含まれる。
# MAGIC * product_quality: 各製品の、実際の品質。
# MAGIC 
# MAGIC ### アーキテクチャ
# MAGIC 
# MAGIC <img src="https://joelcthomas.github.io/modeldrift/img/model_drift_architecture.png" width="1300">
# MAGIC 
# MAGIC <hr>
# MAGIC 
# MAGIC エンドツーエンドのパイプライン。
# MAGIC 
# MAGIC データアクセス -> データ準備 -> モデルの訓練 -> デプロイ -> モニタリング -> アクション & フィードバックのループ

# COMMAND ----------

# MAGIC %md
# MAGIC ## デモ用データの準備

# COMMAND ----------

from pyspark.sql.functions import *

# データを読み込む - センサー測定値
sensor_df = spark.read.format('delta').load('/home/masahiko.kitamura@databricks.com/sensor_reading')

# データを読み込む - Ground Truth(ラベルデータ)
quality_df = spark.read.format('delta').load('/home/masahiko.kitamura@databricks.com/product_quality')

# COMMAND ----------

from pyspark.sql.functions import *

# 指定した期間のデータをロードする処理
def load_df(src_df, datetime_col, start, end):
  return src_df.filter( col(datetime_col).between(start, end) )

# COMMAND ----------

# MAGIC %md
# MAGIC ## データ探索 (今日=2019/07/11)

# COMMAND ----------

# DBTITLE 1,利用可能なデータを確認する
# 時間窓でDFを切り出す (期間: 2019/07/01 - 07/10)
sensor_df_sub = load_df(sensor_df, 'process_time', start='2019-07-01 00:00:00', end='2019-07-10 23:59:00')
quality_df_sub = load_df(quality_df, 'qualitycheck_time', start='2019-07-01 00:00:00', end='2019-07-10 23:59:00')

# 2種類のデータ:
# データ1 => センサ計測データ
display( sensor_df_sub )

# データ2 => ラベルデータ(Ground Truth)
display( quality_df_sub )

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2つのデータを結合させてデータ探索する(EDA)

# COMMAND ----------

# DBTITLE 1,2つのデータを結合させてデータ探索する(EDA)
# 計測データとlabeledデータを結合させる
combined_df = sensor_df_sub.join(quality_df_sub, 'pid')

# 確認
display(combined_df)

# レコード数確認
print( 'レコード数=> ', combined_df.count()  )

# COMMAND ----------

# DBTITLE 1,統計サマリ
display(
  combined_df.summary()
)

# COMMAND ----------

# DBTITLE 1,EDA1 - 時系列
display(
  combined_df.orderBy('process_time')
)

# COMMAND ----------

# DBTITLE 1,EDA2 - ヒストグラム(pressure / quality=0, 1毎)
display(
  combined_df
)

# COMMAND ----------

# DBTITLE 1,EDA3 - Scatter Plot ( quality=0, 1毎 )
display(
  combined_df
)

# COMMAND ----------

# DBTITLE 1,EDA4 - Box plot (duration / quality=0, 1毎 )
display(
  combined_df
)

# COMMAND ----------

# DBTITLE 1,EDA - DataFrameのtemp viewを作成するとSQLが使える
combined_df.createOrReplaceTempView('combined_view')

# COMMAND ----------

# DBTITLE 1,EDA5 - SQLでテーブルを表示
# MAGIC %sql
# MAGIC SELECT * FROM combined_view

# COMMAND ----------

# DBTITLE 1,EDA6 - SQLで日毎のtemp推移を見る
# MAGIC %sql
# MAGIC SELECT
# MAGIC   pdate,
# MAGIC   quality,
# MAGIC   avg(temp),
# MAGIC   avg(pressure),
# MAGIC   avg(duration)
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     date_trunc('DAY', process_time) as pdate
# MAGIC   FROM
# MAGIC     combined_view
# MAGIC )
# MAGIC GROUP BY pdate, quality

# COMMAND ----------

# MAGIC %md
# MAGIC ## 機械学習・モデル作成 (今日=2019/07/11)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forestを使ってモデルを作成する

# COMMAND ----------

# random forest classifierで学習
import mlflow
mlflow.autolog()

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# tain data/ test dataに分ける
(train_df, test_df) = combined_df.randomSplit([0.8, 0.2], seed=42)

# stage0: qualityカラムをindex化
stringIndexer = StringIndexer(inputCols=['quality'], outputCols=['quality_idx'])

# stage1: 説明変数のカラムをベクトル化(一つのカラムにまとめる)
vecAssembler=VectorAssembler(
  inputCols = ['temp', 'pressure', 'duration'],
  outputCol = 'features',
).setHandleInvalid('skip')

# stage2: ランダムフォレスト学習器
rf = RandomForestClassifier(
  labelCol = 'quality_idx', 
  featuresCol = 'features',
  numTrees=20,
  maxDepth=5,
  seed=42
)

# 3つのstageをパイプライン化
pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])

# 学習(fit)
model = pipeline.fit(train_df)

# 予測/スコアリング
pred_df = model.transform(test_df)

# 評価(accuracy)
evaluator = MulticlassClassificationEvaluator(
  labelCol = 'quality_idx',
  predictionCol = 'prediction',
  metricName = 'accuracy'
)

accuracy = evaluator.evaluate(pred_df)
print(f'accuracy => {accuracy}')


# モデルの理解(feature importance)
rf_model = model.stages[-1]
print( list( zip(vecAssembler.getInputCols(), rf_model.featureImportances) ) )


# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflowを用いてモデルをトラックキング

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# confusion matrixを描画する処理
def confusion_matrix_plt(confusion_df, fig_path='confusion_matrix.png'):
  conf_mat = confusion_df.toPandas()
  conf_mat = pd.pivot_table(conf_mat, values='count', index=['prediction'], columns=['quality_idx'], aggfunc=np.sum, fill_value=0)

  plt.clf()
  fig = plt.figure(figsize=(4,4))

  sns.heatmap(conf_mat, annot=True, fmt='d', square=True, cmap='OrRd')
  plt.yticks(rotation=0)
  plt.xticks(rotation=90)
  plt.savefig(fig_path)
  plt.show()

# COMMAND ----------

import mlflow
import mlflow.spark

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



with mlflow.start_run(run_name='first_random_forest') as run:

  # stage0: qualityカラムをindex化
  stringIndexer = StringIndexer(inputCols=['quality'], outputCols=['quality_idx'])

  # stage1: 説明変数のカラムをベクトル化(一つのカラムにまとめる)
  vecAssembler=VectorAssembler( inputCols = ['temp', 'pressure', 'duration'], outputCol = 'features').setHandleInvalid('skip')

  # stage2: ランダムフォレスト学習器
  rf = RandomForestClassifier( labelCol = 'quality_idx', featuresCol = 'features', numTrees=20, maxDepth=5, seed=42)

  # 3つのstageをパイプライン化
  pipeline = Pipeline( stages=[stringIndexer, vecAssembler, rf] )

  # 学習(fit)
  model = pipeline.fit(train_df)

  # 予測/スコアリング
  pred_df = model.transform(test_df)

  # 評価(accuracy)
  evaluator = MulticlassClassificationEvaluator( labelCol = 'quality_idx', predictionCol = 'prediction', metricName = 'accuracy')
  accuracy = evaluator.evaluate(pred_df)
  print(f'accuracy => {accuracy}')

  # confusion matrix
  confusion_df = pred_df.select('prediction', 'quality_idx').groupBy('prediction', 'quality_idx').count()
  confusion_matrix_plt(confusion_df)

  
  # モデルの理解(feature importance)
  print( list( zip(vecAssembler.getInputCols(), rf_model.featureImportances) ) )

  
  # mlflowでトラッキング
  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_param('numTrees', 20)
  mlflow.log_param('maxDepth', 5)
  mlflow.spark.log_model(model, 'spark-model')
  
  mlflow.log_artifact('confusion_matrix.png')

  

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLFlowとGrid Search

# COMMAND ----------

# DBTITLE 0,ハイパーパラメータチューニング、Gridサーチ、モデル比較用に一般化・関数化
import mlflow
import mlflow.spark

def train_model(run_name, stages, params):
  with mlflow.start_run(run_name='first_random_forest') as run:

    alg = stages[-1]
    alg.setParams(**params)
    
    pipeline = Pipeline( stages=stages )

    # 学習(fit)
    model = pipeline.fit(train_df)

    # 予測/スコアリング
    pred_df = model.transform(test_df)

    # 評価(accuracy)
    evaluator = MulticlassClassificationEvaluator( labelCol = 'quality_idx', predictionCol = 'prediction', metricName = 'accuracy')
    accuracy = evaluator.evaluate(pred_df)
    print(f'accuracy => {accuracy}')

    # confusion matrix
    confusion_df = pred_df.select('prediction', 'quality_idx').groupBy('prediction', 'quality_idx').count()
    confusion_matrix_plt(confusion_df)

    # mlflowでトラッキング
    for k,v in params.items():
      mlflow.log_param(k, v)
      
    mlflow.log_metric('accuracy', accuracy)
    mlflow.spark.log_model(model, str(alg))
    mlflow.log_artifact('confusion_matrix.png')



# COMMAND ----------

# 前処理
stringIndexer = StringIndexer(inputCols=['quality'], outputCols=['quality_idx'])
vecAssembler=VectorAssembler( inputCols = ['temp', 'pressure', 'duration'], outputCol = 'features').setHandleInvalid('skip')

# ランダムフォレストのインスタンス
rf = RandomForestClassifier( labelCol = 'quality_idx', featuresCol = 'features')

# パイプライン
stages=[stringIndexer, vecAssembler, rf]

# ハイパーパラメータ
params={
  'numTrees': 20,
  'maxDepth': 5,
  'seed': 42
}

# とりあえず一つのハイパーパラメータを設定して、モデルを作成する
train_model('second_random_forest', stages, params)

# COMMAND ----------

# DBTITLE 0,マニュアルでGrid Search
# マニュアルでGridサーチを実施する
numTreesList = [10, 25]
maxDepthList = [3, 10]

for num_trees in numTreesList:
  for max_depth in maxDepthList:
    params={ 'numTrees': num_trees, 'maxDepth': max_depth, 'seed': 42}
    train_model('manual_random_forest', stages, params)


# COMMAND ----------

# MAGIC %md
# MAGIC ### AutoML的に GridSearch + Cross validationを実施する

# COMMAND ----------

# grid search + cross validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, IndexToString, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# 前処理、ランダムフォレストのパイプライン
stringIndexer = StringIndexer(inputCols=['quality'], outputCols=['quality_idx'])
vecAssembler=VectorAssembler( inputCols = ['temp', 'pressure', 'duration'], outputCol = 'features').setHandleInvalid('skip')
rf = RandomForestClassifier( labelCol = 'quality_idx', featuresCol = 'features')
stages=[stringIndexer, vecAssembler, rf]

# Gridサーチの範囲を決める
paramGrid = (
  ParamGridBuilder()
  .addGrid(rf.maxDepth, list(range(3,7)) )
  .addGrid(rf.numTrees, list(range(5,15)) )
  .build()
)

# モデルの評価関数を定義する
evaluator = MulticlassClassificationEvaluator( labelCol = 'quality_idx', predictionCol = 'prediction', metricName = 'accuracy')

# クロスバリデーション + Gridサーチを設定する
cv = CrossValidator(
  estimator=Pipeline( stages=stages ),
  evaluator=MulticlassClassificationEvaluator( labelCol = 'quality_idx', predictionCol = 'prediction', metricName = 'accuracy'),
  estimatorParamMaps=paramGrid,
  numFolds=3,
  seed=42
)


# COMMAND ----------

# tain data/ test dataに分ける
(train_df, test_df) = combined_df.randomSplit([0.8, 0.2], seed=42)

# mlflowでトラック
with mlflow.start_run():
  
  # 並列数=8でGridサーチ、クロスバリデーションを実施する
  cvModel = cv.setParallelism(8).fit(train_df)

  # BestなモデルをMLflowでトラック
  test_metric = evaluator.evaluate(cvModel.transform(test_df))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)
  mlflow.log_param('model_date', '2019-07-11')
  mlflow.spark.log_model(spark_model=cvModel.bestModel, artifact_path='best_model')


# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのデプロイ (今日=2019/07/11)
# MAGIC 
# MAGIC MLflowのモデルレジストリ機能を使って、モデルをプロダクションにデプロイ
# MAGIC 
# MAGIC 1. notebookに紐づいているMLflowを開く
# MAGIC 1. 上記の`best_model`で登録したモデルをモデル名`glassware-quality`でレジストリに登録する(`ver.1`としてcommitされる)
# MAGIC 1. (動作テスト、ステージングテストが完了したとして)
# MAGIC 1. Productionにプッシュする(デプロイ)

# COMMAND ----------

# Productionのモデルをロードする(デプロイ)
prod_model = mlflow.spark.load_model('models:/glassware-quality/production')

# COMMAND ----------

# MAGIC %md
# MAGIC ### ドリフトのモニタリング(開始日=2019/07/11, 今日=2019/07/23)

# COMMAND ----------

from pyspark.sql import Window

# モデル性能の移動平均の算出処理
def monitor_df(pred_df):
  summary_df = ( 
    pred_df
    .join(quality_df, 'pid')
    .withColumn('is_accurate_prediction', when( col('quality_idx') == col('prediction'), 1 ).otherwise(0)  )
    .groupBy( window(col('process_time'), '1 day').alias('window'), col('is_accurate_prediction') )
    .count()
    .withColumn('window_day', expr('to_date(window.start)') )
    .withColumn('total', sum( col('count') ).over(Window.partitionBy('window_day'))  )
    .withColumn('ratio', col('count') / col('total') * 100.0 )
    .select('window_day', 'is_accurate_prediction', 'count', 'total', 'ratio')
    .withColumn('is_accurate_prediction', when( col('is_accurate_prediction') == 1, 'Accurate').otherwise('Inaccurate'))
    .orderBy('window_day')
  )
  return summary_df

# COMMAND ----------

# 日数が経ち、新しいデータが利用可能になる(07/11 - 07/23)
sensor_df_sub = load_df(sensor_df, 'process_time', start='2019-07-11 00:00:00', end='2019-07-22 23:59:00')
quality_df_sub = load_df(quality_df, 'qualitycheck_time', start='2019-07-11 00:00:00', end='2019-07-22 23:59:00')
combined_df = sensor_df_sub.join(quality_df_sub, 'pid')

# モデルを使って推定(スコアリング)を実施
pred_df = prod_model.transform(combined_df)

# 結果を確認
display(
  pred_df
)

# COMMAND ----------

summary_df_2019_07_11 = monitor_df(pred_df)
display(summary_df_2019_07_11)

# COMMAND ----------

# MAGIC %md
# MAGIC 今日は7/23だが、ラベルデータは7/20分までしか利用可能になっていない。
# MAGIC 
# MAGIC 7/19の結果に関して、ドリフトを検知。ドリフトの検知は3日間程度のラグが発生する。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 再学習(今日=2019/07/23)

# COMMAND ----------

# 時間窓でDFを切り出す
sensor_df_sub = load_df(sensor_df, 'process_time', start='2019-07-18 00:00:00', end='2019-07-23 00:00:00')
quality_df_sub = load_df(quality_df, 'qualitycheck_time', start='2019-07-18 00:00:00', end='2019-07-23 00:00:00')

# 計測データとlabeledデータを結合させる
combined_df = sensor_df_sub.join(quality_df_sub, 'pid')

# 学習データ、テストデータの分離
(train_df, test_df) = combined_df.randomSplit([0.8, 0.2], seed=42)

# 再学習
with mlflow.start_run():
  
  # 並列数=8でGridサーチ、クロスバリデーションを実施する
  cvModel = cv.setParallelism(8).fit(train_df)

  # Bestなモデルを取得する
  #best_model_2019_07_11 = cvModel.bestModel
  test_metric = evaluator.evaluate(cvModel.transform(test_df))
  mlflow.log_metric('test_' + evaluator.getMetricName(), test_metric)
  mlflow.log_param('model_date', '2019-07-23')
  mlflow.spark.log_model(spark_model=cvModel.bestModel, artifact_path='best_model')

# cvModel = cv.setParallelism(8).fit(train_df)

# # Bestなモデルを取得する
# best_model_2019_07_23 = cvModel.bestModel

# デプロイ

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのデプロイ2 (今日=2019/07/23)
# MAGIC 
# MAGIC 1. 前回と同様に、`best_model`で登録したモデルをモデル名`glassware-quality`でレジストリに登録する(`ver.2`としてcommitされる)
# MAGIC 1. (動作テスト、ステージングテストが完了したとして)
# MAGIC 1. Productionにプッシュする(デプロイ)

# COMMAND ----------

# Productionのモデルをロードする(デプロイ)
prod_model = mlflow.spark.load_model('models:/glassware-quality/production')

# COMMAND ----------

# MAGIC %md
# MAGIC ## ドリフトのモニタリング2 (開始日=2019/07/23, 今日=2019/08/07)

# COMMAND ----------

# 時間窓でDFを切り出す
sensor_df_sub = load_df(sensor_df, 'process_time', start='2019-07-23 00:00:00', end='2019-08-07 00:00:00')
quality_df_sub = load_df(quality_df, 'qualitycheck_time', start='2019-07-23 00:00:00', end='2019-08-07 00:00:00')
combined_df = sensor_df_sub.join(quality_df_sub, 'pid')


# モデルを使って推定(スコアリング)を実施
pred_df = prod_model.transform(combined_df)

# モニタリング
summary_df_2019_07_23 = monitor_df(pred_df)
display(summary_df_2019_07_23)

# COMMAND ----------

# MAGIC %md
# MAGIC ### もし古いモデルをそのまま使い続けていたら....

# COMMAND ----------

# 全期間のデータ
sensor_df_sub = load_df(sensor_df, 'process_time', start='2019-07-11 00:00:00', end='2019-08-07 00:00:00')
quality_df_sub = load_df(quality_df, 'qualitycheck_time', start='2019-07-11 00:00:00', end='2019-08-07 00:00:00')
combined_df = sensor_df_sub.join(quality_df_sub, 'pid')

# 古いモデルで推定を実施したら、という仮定
previous_model = mlflow.spark.load_model('models:/glassware-quality/1') # ver.1のモデルをload
pred_df = previous_model.transform(combined_df)
summary_df_2019_07_11 = monitor_df(pred_df)

display(
  summary_df_2019_07_11.withColumn('model_name', lit('model_2019_07_11'))
  .union(
    summary_df_2019_07_23.withColumn('model_name', lit('model_2019_07_23'))
  )



)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Appendix: Databricksの通知機能の実装について
# MAGIC 
# MAGIC Pythonのrequestsライブラリを使用して、Webhookとの連携が可能です。これにより、Slackをはじめとするメッセージングサービスとも連携が可能です。
# MAGIC 
# MAGIC サンプルコード:
# MAGIC ```
# MAGIC def postToAPIEndpoint(content, webhook=""):
# MAGIC     """
# MAGIC     Post message to Teams to log progress
# MAGIC     """
# MAGIC     import requests
# MAGIC     from requests.exceptions import MissingSchema
# MAGIC     from string import Template
# MAGIC 
# MAGIC     t = Template('{"text": "${content}"}')
# MAGIC 
# MAGIC     try:
# MAGIC         response = requests.post(
# MAGIC             webhook,
# MAGIC             data=t.substitute(content=content),
# MAGIC             headers={"Content-Type": "application/json"},
# MAGIC         )
# MAGIC         return response
# MAGIC     except MissingSchema:
# MAGIC         print(
# MAGIC             "Please define an appropriate API endpoint use by defining the `webhook` argument"
# MAGIC         )
# MAGIC 
# MAGIC 
# MAGIC postToAPIEndpoint("This is my post from Python", webhookMLProductionAPIDemo)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC #END
