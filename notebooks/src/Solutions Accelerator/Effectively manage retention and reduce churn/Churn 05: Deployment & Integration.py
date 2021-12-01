# Databricks notebook source
# MAGIC %md このノートブックの目的は、学習したモデルを使って、下流のCRMシステムにインポート可能な予測を生成することです。 このノートブックは、Databricks ML 7.1+と**CPUベース**のノードを利用したクラスタ上で実行する必要があります。

# COMMAND ----------

# MAGIC %md ###Step 1: スコアリングのためのデータの取得
# MAGIC 
# MAGIC 解約予測モデルを学習する目的は、積極的なリテンション管理を行う対象顧客を特定することです。そのためには、定期的に特徴情報から予測を行い、その予測をキャンペーンをサポートするシステムで利用できるようにする必要があります。
# MAGIC 
# MAGIC そこで今回は、最近学習したモデルを取り出して、SalesforceやMicrosoft Dynamicsなど、カスタムデータのインポートが可能な多くのシステムにインポートできるよう、スコアリングされた出力を生成する方法を検討します。 このような出力をこれらのシステムに統合するには複数の方法がありますが、ここでは最もシンプルな方法、すなわち*フラットファイルへのエクスポートを検討します。
# MAGIC 
# MAGIC まず最初に、予測を行う期間に関連する特徴データを取得します。 2017年2月のデータでモデルを学習し、2017年3月のデータでモデルを評価したことを考えると、2017年4月の予測出力を生成することは理にかなっています。 とはいえ、このデータセットに関連するKaggleコンペティションの爪先を踏むことは避けたいので、ここでは2017年3月の予測出力を生成することに限定します。
# MAGIC 
# MAGIC これまでのノートブックとは異なり、データの取得はフィーチャーと顧客識別子に限定し、実際に将来の予測を行う場合には解約テーブルは必要ないので無視します。データをまずSpark DataFrameにロードし、次にpandas dataframeにロードすることで、出力を生成するための2つの異なる手法を実演しますが、それぞれの手法は異なるデータフレームタイプに依存します。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
import mlflow
import mlflow.pyfunc

import pandas as pd
import shutil, os

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import struct

# COMMAND ----------

# DBTITLE 1,特徴量とCustomer IDの読み込み
# retrieve features & identifier to Spark DataFrame
input = spark.sql('''
  SELECT
    a.*,
    b.days_total,
    b.days_with_session,
    b.ratio_days_with_session_to_days,
    b.days_after_exp,
    b.days_after_exp_with_session,
    b.ratio_days_after_exp_with_session_to_days_after_exp,
    b.sessions_total,
    b.ratio_sessions_total_to_days_total,
    b.ratio_sessions_total_to_days_with_session,
    b.sessions_total_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp,
    b.ratio_sessions_total_after_exp_to_days_after_exp_with_session,
    b.seconds_total,
    b.ratio_seconds_total_to_days_total,
    b.ratio_seconds_total_to_days_with_session,
    b.seconds_total_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp,
    b.ratio_seconds_total_after_exp_to_days_after_exp_with_session,
    b.number_uniq,
    b.ratio_number_uniq_to_days_total,
    b.ratio_number_uniq_to_days_with_session,
    b.number_uniq_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp,
    b.ratio_number_uniq_after_exp_to_days_after_exp_with_session,
    b.number_total,
    b.ratio_number_total_to_days_total,
    b.ratio_number_total_to_days_with_session,
    b.number_total_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp,
    b.ratio_number_total_after_exp_to_days_after_exp_with_session
  FROM kkbox.test_trans_features a
  INNER JOIN kkbox.test_act_features b
    ON a.msno=b.msno
  ''')

# extract features to pandas DataFrame
input_pd = input.toPandas()
X = input_pd.drop(['msno'], axis=1) # features for making predictions
msno = input_pd[['msno']] # customer identifiers to which we will append predictions

# COMMAND ----------

# MAGIC %md ###Step 2a: 現状のモデルを利用した予測出力の生成
# MAGIC 
# MAGIC [Microsoft Dynamics CRM Common Data Service](https://docs.microsoft.com/en-us/powerapps/developer/common-data-service/import-data)と[Salesforce DataLoader](https://developer.salesforce.com/docs/atlas.en-us.dataLoader.meta/dataLoader/data_loader.htm)のどちらを使用するかに関わらず、ヘッダー行を含むUTF-8で区切られたテキストファイルを生成する必要があります。pandasのデータフレームとモデルのネイティブ機能を利用して、次のようにしてこのようなファイルを提供することができます。

# COMMAND ----------

# DBTITLE 1,レジストリからモデルを取得する
model_name = 'churn-ensemble'

model = mlflow.pyfunc.load_model(
  'models:/{0}/production'.format(model_name)
  )

# COMMAND ----------

# DBTITLE 1,予測値をファイルに保存
# databricks location for the output file
output_path = '/mnt/kkbox/output_native/'
shutil.rmtree('/dbfs'+output_path, ignore_errors=True) # delete folder & contents if exists
dbutils.fs.mkdirs(output_path) # recreate folder

# generate predictions
y_prob = model.predict(X)

# assemble output dataset
output = pd.concat([
    msno, 
    pd.DataFrame(y_prob, columns=['churn'])
    ], axis=1
  )
output['period']='2017-03-01'

# write output to file
output[['msno', 'period', 'churn']].to_csv(
  path_or_buf='/dbfs'+output_path+'output.txt', # use /dbfs fuse mount to access cloud storage
  sep='\t',
  header=True,
  index=False,
  encoding='utf-8'
  )

# COMMAND ----------

# MAGIC %md ファイルとその内容を見てみましょう。

# COMMAND ----------

# DBTITLE 1,Examine Output File Contents
print(
  dbutils.fs.head(output_path+'output.txt')
  )

# COMMAND ----------

# MAGIC %md ###Step 2b: Spark UDFを使って予測出力を生成する
# MAGIC 
# MAGIC ステップ2aのようにsklearnモデルのネイティブAPIを使用して予測を行うことは、多くのデータサイエンティストにとって馴染みのあることですが、このフェーズを実施するのは一般的にはデータエンジニアです。 このような人たちにとっては、Spark SQLを使う方が親しみやすいかもしれませんし、大量の顧客を相手にするシナリオでは拡張性も高いでしょう。
# MAGIC 
# MAGIC モデルをSparkで使用するには、まずユーザー定義関数（UDF）として登録する必要があります。

# COMMAND ----------

# DBTITLE 1,モデルをUDFに登録
churn_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'models:/{0}/production'.format(model_name), 
  result_type = DoubleType()
  )

# COMMAND ----------

# MAGIC %md それでは、UDFを使って予測を生成してみましょう。 UDFをSQL文の中で使用することも可能ですが、今回は関数に渡す必要のある特徴のリストが非常に長いため、Spark SQLの*struct*型を使用して特徴フィールドを結合します。 これにより、長い機能のリストを簡単に渡すことができ、機能数が増えても将来の変更を最小限に抑えることができます。

# COMMAND ----------

# DBTITLE 1,予測値を含むデータフレームの生成
output_path = '/mnt/kkbox/output_spark/'

# get list of columns in dataframe
input_columns = input.columns

# assemble struct containing list of features
features = struct([feature_column for feature_column in input_columns if feature_column != 'msno']) 

# generate output dataset 
output = (
  input
    .withColumn(
      'churn', 
      churn_udf( features )  # call udf to generate prediction
      )
    .selectExpr('msno', '\'2017-03-01\' as period', 'churn')
  )

# write output to storage
(output
    .repartition(1)  # repartition to generate a single output file
    .write
    .mode('overwrite')
    .csv(
      path=output_path,
      sep='\t',
      header=True,
      encoding='UTF-8'
      )
  )

# COMMAND ----------

# MAGIC %md 上のセルでは、データを出力フォルダに書き込んでいることに注目してください。 *repartition()*メソッドを使用して、データが1つのファイルに書き込まれるようにすることはできますが、ここで生成されるファイルの名前を直接制御することはできません。

# COMMAND ----------

# DBTITLE 1,出力フォルダの内容を確認する
display(
  dbutils.fs.ls(output_path)
       )

# COMMAND ----------

# MAGIC %md ファイル名が重要な場合は、Pythonのネイティブ機能を使って、CSV出力ファイルの名前を後から変更することができます。

# COMMAND ----------

# DBTITLE 1,出力ファイルの名前変更
for file in os.listdir('/dbfs'+output_path):
  if file[-4:]=='.csv':
    shutil.move('/dbfs'+output_path+file, '/dbfs'+output_path+'output.txt' )

# COMMAND ----------

# MAGIC %md このファイルの内容を見てみると、先に生成した内容と同じですが、顧客のソート順は異なるかもしれません（どちらのステップでもソートは指定されていないため）。

# COMMAND ----------

# DBTITLE 1,出力ファイルの内容を確認する
print(
  dbutils.fs.head(output_path+'output.txt')
  )
