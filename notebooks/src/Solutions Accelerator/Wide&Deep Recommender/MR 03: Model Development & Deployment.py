# Databricks notebook source
# MAGIC %md このノートブックの目的は、前のノートブックで設計された機能を使って、ワイドでディープな協調フィルタレコメンダーを訓練、評価、導入することです。 このノートブックは **Databricks 8.1+ ML cluster** で実行する必要があります。

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
import pyspark.sql.functions as f
from pyspark.sql.types import *

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from petastorm.spark import SparkDatasetConverter, make_spark_converter
from petastorm import TransformSpec

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import mlflow
from mlflow.tracking import MlflowClient

import platform

import numpy as np
import pandas as pd

import datetime
import os
import requests

# COMMAND ----------

# MAGIC %md ## Step 1: データの準備
# MAGIC 
# MAGIC 前回のノートでは、ユーザーと商品の両方の特徴量と、特定のユーザーと商品の組み合わせが学習期間中に購入されたかどうかを示すラベルが用意されていました。 ここでは、これらのデータを取得し、組み合わせてモデルに入力します。

# COMMAND ----------

# DBTITLE 1,特徴量とラベルを取得する
# retrieve features and labels
product_features = spark.table('instacart.product_features')
user_features = spark.table('instacart.user_features')
labels = spark.table('instacart.labels')

# assemble full feature set
labeled_features = (
  labels
  .join(product_features, on='product_id')
  .join(user_features, on='user_id')
  )

# display results
display(labeled_features)

# COMMAND ----------

# MAGIC %md 多数の特徴量があるため、フィールドのメタデータを取得する必要があります。 このメタデータは、後のステップでデータ入力を設定するのに役立ちます。

# COMMAND ----------

# DBTITLE 1,特徴量とラベルを取り込む
# identify label column
label_col = 'label'

# identify categorical feature columns
cat_features = ['aisle_id','department_id','user_id','product_id']

# capture keys for each of the categorical feature columns
cat_keys={}
for col in cat_features:
  cat_keys[col] = (
    labeled_features
      .selectExpr('{0} as key'.format(col))
      .distinct()
      .orderBy('key')
      .groupBy()
        .agg(f.collect_list('key').alias('keys'))
      .collect()[0]['keys']
    )

# all other columns (except id) are continous features
num_features = labeled_features.drop(*(['id',label_col]+cat_features)).columns

# COMMAND ----------

# MAGIC %md ここで、データをトレーニングセット、検証セット、テストセットに分割します。 ここでは、動的に分割するのではなく、事前に分割することで、ラベルの層別サンプルを実行することができます。 層化サンプルを使用することで、データを分割した際に、存在感の薄いポジティブクラス（トレーニング期間中に特定の製品を購入したことを示す）が一貫して存在するようになります。

# COMMAND ----------

# DBTITLE 1,ポジティブなクラスの表現を評価する
instance_count = labeled_features.count()
positive_count = labels.filter(f.expr('label=1')).count()

print('{0:.2f}% positive class across {1} instances'.format(100 * positive_count/instance_count, instance_count))

# COMMAND ----------

# DBTITLE 1,データをトレーニング、バリデーション、テストに分割
# fraction to hold for training
train_fraction = 0.6

# sample data, stratifying on labels, for training
train = (
  labeled_features
    .sampleBy(label_col, fractions={0: train_fraction, 1: train_fraction})
  )

# split remaining data into validation & testing datasets (with same stratification)
valid = (
  labeled_features
    .join(train, on='id', how='leftanti') # not in()
    .sampleBy(label_col, fractions={0:0.5, 1:0.5})
  )

test = (
  labeled_features
    .join(train, on='id', how='leftanti') # not in()
    .join(valid, on='id', how='leftanti') # not in()
  )

# COMMAND ----------

# MAGIC %md トレーニング、検証、テストのデータセットは現在Spark Dataframeとして存在しており、かなりの大きさになる可能性があります。 データをpandas Dataframeに変換するとout of memoryエラーが発生する可能性があるため、代わりにSpark Dataframeを[Petastorm](https://petastorm.readthedocs.io/en/latest/)データセットに変換します。PetastormはSparkのデータをParquetにキャッシュし、TensorflowやPyTorchなどのライブラリにそのデータへの高速なバッチアクセスを提供するライブラリです。
# MAGIC 
# MAGIC **注意** Petastormは、キャッシュされたファイルが小さすぎると訴えることがあります。 repartition()メソッドを使用して、各データセットで生成されるキャッシュファイルの数を調整しますが、シナリオに応じて最適なファイル数を決定するために数を調整してください。

# COMMAND ----------

# DBTITLE 1,データをキャッシュしてアクセスを高速化
# configure temp cache for petastorm files
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:///dbfs/tmp/instacart/pstorm_cache') # the file:// prefix is required by petastorm

# persist dataframe data to petastorm cache location
train_pstorm = make_spark_converter(train.repartition(4))  
valid_pstorm = make_spark_converter(valid.repartition(4)) 
test_pstorm = make_spark_converter(test.repartition(4)) 

# COMMAND ----------

# MAGIC %md Petastormキャッシュのデータにアクセスできるようにするためには、データを読み込んでTensorflowが期待するフォーマットに変換するスペックを定義する必要がある。 このフォーマットでは、特徴が辞書として提示され、ラベルがスカラー値として提示される必要がある。

# COMMAND ----------

# DBTITLE 1,データスペックの定義
def get_data_specs(epochs=1, batch_size=128):
  
  # define functions to transform data into req'ed format
  def get_input_fn(dataset_context_manager):
    
    # re-structure a row as ({features}, label)
    def _to_tuple(row): 
      features = {}
      for col in cat_features + num_features:
        features[col] = getattr(row, col)
      return features, getattr(row, label_col)
    
    def fn(): # called by estimator to perform row structure conversion
      return dataset_context_manager.__enter__().map(_to_tuple)
    
    return fn

  # access petastorm cache as tensorflow dataset
  train_ds = train_pstorm.make_tf_dataset(batch_size=batch_size)
  valid_ds = valid_pstorm.make_tf_dataset()
  
  # define spec to return transformed data for model training & evaluation
  train_spec = tf.estimator.TrainSpec(
                input_fn=get_input_fn(train_ds), 
                max_steps=int( (train_pstorm.dataset_size * epochs) / batch_size )
                )
  eval_spec = tf.estimator.EvalSpec(
                input_fn=get_input_fn(valid_ds)
                )
  
  return train_spec, eval_spec

# COMMAND ----------

# MAGIC %md 以下のように行を取得することで、仕様を確認することができます。 なお、training（最初の）specのデフォルトのバッチサイズは128レコードです。

# COMMAND ----------

# DBTITLE 1,Specを確認する
# retrieve specs
specs = get_data_specs()

# retrieve first batch from first (training) spec
next(
  iter(
    specs[0].input_fn().take(1)
    )
  )

# COMMAND ----------

# MAGIC %md ## Step 2: モデルを定義する
# MAGIC 
# MAGIC データが揃ったところで、ワイド＆ディープモデルを定義します。 このために、これらの種類のモデルの定義を簡略化する[Tensorflow's DNNLinearCombinedClassifier estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNLinearCombinedClassifier)を利用します。
# MAGIC 
# MAGIC DNNLinearCombinedClassifier estimatorの特徴入力は、*ワイド*な線形モデルに関連するものと、*ディープ*なニューラルネットワークに関連するものに分けられる。 ワイドモデルへの入力は、ユーザーIDと製品IDの組み合わせです。 このようにして、線形モデルは、どのユーザーがどの製品を購入したかを記憶するように訓練されます。 これらの特徴は、単純なカテゴリー特徴[順序値によって識別される](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_identity)、または[より少ない数のバケットにハッシュ化される](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket)としてモデルに導入することができます。 ユーザー製品の[クロスハッシュ](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column)を含めることで、モデルはユーザーと製品の組み合わせをよりよく理解することができます。
# MAGIC 
# MAGIC **注** 後続のロジックの多くは、関数でカプセル化されています。 これにより、ノートブックの後半で発生する分散処理を簡単に実装することができ、ほとんどのTensorflowの実装ではかなり標準的なものです。

# COMMAND ----------

# DBTITLE 1,Wide Featuresを定義する
def get_wide_features():

  wide_columns = []

  # user_id
  #wide_columns += [tf.feature_column.categorical_column_with_identity(
  #    key='user_id', 
  #    num_buckets=np.max(cat_keys['user_id'])+1 # create one bucket for each value from 0 to max
  #    )]
  wide_columns += [
    tf.feature_column.categorical_column_with_hash_bucket(
       key='user_id', 
       hash_bucket_size=1000,
       dtype=tf.dtypes.int64# create one bucket for each value from 0 to max
       )]

  # product_id
  #wide_columns += [
  #  tf.feature_column.categorical_column_with_identity(
  #    key='product_id', 
  #    num_buckets=np.max(cat_keys['product_id'])+1 # create one bucket for each value from 0 to max
  #    )]
  wide_columns += [
    tf.feature_column.categorical_column_with_hash_bucket(
       key='product_id', 
       hash_bucket_size=100,
       dtype=tf.dtypes.int64 # create one bucket for each value from 0 to max
       )]

  # user-product cross-column (set column spec to ensure presented as int64)
  wide_columns += [
    tf.feature_column.crossed_column(
      [ tf.feature_column.categorical_column_with_identity(key='user_id', num_buckets=np.max(cat_keys['user_id'])+1),
        tf.feature_column.categorical_column_with_identity(key='product_id', num_buckets=np.max(cat_keys['product_id'])+1)
        ], 
      hash_bucket_size=1000
      )] 

  return wide_columns

# COMMAND ----------

# MAGIC %md モデルのディープ（ニューラルネットワーク）コンポーネントの特徴入力は、ユーザーや製品をより一般的な方法で表現する特徴です。特定のユーザーや製品のIDを避けることで、ディープモデルはユーザーや製品間の好みを示す属性を学習します。カテゴリカルな特徴については、[embedding](https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column)を使用して、特徴データを簡潔に捉えます。 エンベッディングの次元数は、[this tutorial](https://tensorflow2.readthedocs.io/en/stable/tensorflow/g3doc/tutorials/wide_and_deep/)のガイダンスに基づいています。

# COMMAND ----------

# DBTITLE 1,Deep Featuresを定義する
def get_deep_features():
  
  deep_columns = []

  # categorical features
  for col in cat_features:

    # don't use user ID or product ID
    if col not in ['user_id','product_id']:

      # base column definition
      col_def = tf.feature_column.categorical_column_with_identity(
        key=col, 
        num_buckets=np.max(cat_keys[col])+1 # create one bucket for each value from 0 to max
        )

      # define embedding on base column def
      deep_columns += [tf.feature_column.embedding_column(
                          col_def, 
                          dimension=int(np.max(cat_keys[col])**0.25)
                          )] 

  # continous features
  for col in num_features:
    deep_columns += [tf.feature_column.numeric_column(col)]  
    
  return deep_columns

# COMMAND ----------

# MAGIC %md 機能が定義されたので、推定器を組み立てることができます。
# MAGIC 
# MAGIC **NOTE** オプティマイザーは、[こちら](https://stackoverflow.com/questions/58108945/cannot-do-incremental-training-with-dnnregressor)で指摘された問題に対処するため、クラスとして渡されます。

# COMMAND ----------

# DBTITLE 1,モデルの構築
def get_model(hidden_layers, hidden_layer_nodes_initial_count, hidden_layer_nodes_count_decline_rate, dropout_rate):  
  
  # determine hidden_units structure
  hidden_units = [None] * int(hidden_layers)
  for i in range(int(hidden_layers)):
    # decrement the nodes by the decline rate
    hidden_units[i] = int(hidden_layer_nodes_initial_count * (hidden_layer_nodes_count_decline_rate**i))
 
  # get features
  wide_features = get_wide_features()
  deep_features = get_deep_features()
    
  # define model
  estimator = tf.estimator.DNNLinearCombinedClassifier(
    linear_feature_columns=wide_features,
    linear_optimizer=tf.keras.optimizers.Ftrl,
    dnn_feature_columns=deep_features,
    dnn_hidden_units=hidden_units,
    dnn_dropout=dropout_rate,
    dnn_optimizer=tf.keras.optimizers.Adagrad
    )

  return estimator

# COMMAND ----------

# MAGIC %md ## Step 3: モデルのチューニング
# MAGIC 
# MAGIC モデルをチューニングするためには，評価指標を定義する必要があります． デフォルトでは、DNNLinearCombinedClassifierは、[softmax (categorical) cross entropy](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)メトリックを最小化するようにしています。このメトリックは、予測されたクラス確率と実際のクラスラベルの間の距離を調べます。 (このメトリックは、より正確で自信のあるクラス予測を求めるものと考えることができます。)
# MAGIC 
# MAGIC この指標に基づいてモデルをチューニングしますが、最終結果の評価を助けるために、より伝統的な指標を提供するのも良いかもしれません。 製品を選択される可能性が高いものから低いものの順に提示することを目的としたレコメンダーでは、[mean average precision @ k (MAP@K)](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf)がよく使われます。 この指標は、上位*k*個のレコメンデーションに関連する平均精度を調べるものです。 MAP@Kの値が1.0に近ければ近いほど、それらのレコメンデーションが顧客の製品選択とよりよく一致していることを意味します。
# MAGIC 
# MAGIC MAP@Kを計算するために、私たちは[NVIDIAが提供するコードを再利用](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/WideAndDeep/utils/metrics.py)しており、広告配置のための広くて深いレコメンデーションの実装を行っています。

# COMMAND ----------

# DBTITLE 1,MAP@K Metricの定義
# Adapted from: https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/Recommendation/WideAndDeep/utils/metrics.py
def map_custom_metric(features, labels, predictions):
  
  user_ids = tf.reshape(features['user_id'], [-1])
  predictions = predictions['probabilities'][:, 1]

  # sort user IDs 
  sorted_ids = tf.argsort(user_ids)
  
  # resort values to align with sorted user IDs
  user_ids = tf.gather(user_ids, indices=sorted_ids)
  predictions = tf.gather(predictions, indices=sorted_ids)
  labels = tf.gather(labels, indices=sorted_ids)

  # get unique user IDs in dataset
  _, user_ids_idx, user_ids_items_count = tf.unique_with_counts(
      user_ids, 
      out_idx=tf.int64
      )
  
  # remove any user duplicates
  pad_length = 30 - tf.reduce_max(user_ids_items_count)
  pad_fn = lambda x: tf.pad(x, [(0, 0), (0, pad_length)])
  preds = tf.RaggedTensor.from_value_rowids(
      predictions, user_ids_idx).to_tensor()
  labels = tf.RaggedTensor.from_value_rowids(
      labels, user_ids_idx).to_tensor()
  labels = tf.argmax(labels, axis=1)

  # calculate average precision at k
  return {
      'map@k': tf.compat.v1.metrics.average_precision_at_k(
          predictions=pad_fn(preds),
          labels=labels,
          k=10,
          name="streaming_map")
        }

# COMMAND ----------

# MAGIC %md これで、すべてのロジックをまとめてモデルを定義することができます。

# COMMAND ----------

# DBTITLE 1,トレーニングと評価のロジックの定義
def train_and_evaluate_model(hparams):
  
  # retrieve the basic model
  model = get_model(
    hparams['hidden_layers'], 
    hparams['hidden_layer_nodes_initial_count'], 
    hparams['hidden_layer_nodes_count_decline_rate'], 
    hparams['dropout_rate']
    )
  
  # add map@k metric
  model = tf.estimator.add_metrics(model, map_custom_metric)
  
  # retrieve data specs
  train_spec, eval_spec = get_data_specs( int(hparams['epochs']), int(hparams['batch_size']))
  
  # train and evaluate
  results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
  
  # return loss metric
  return {'loss': results[0]['loss'], 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ここで、モデルを試運転して、すべての可動部が機能しているかどうかを確認します。

# COMMAND ----------

# DBTITLE 1,テストランの実行
hparams = {
  'hidden_layers':2,
  'hidden_layer_nodes_initial_count':100,
  'hidden_layer_nodes_count_decline_rate':0.5,
  'dropout_rate':0.25,
  'epochs':1,
  'batch_size':128
  }

train_and_evaluate_model(hparams)

# COMMAND ----------

# MAGIC %md テストランが成功したので、今度はモデルのハイパーパラメータチューニングを行ってみましょう。[hyperopt](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)を使って、この作業に必要な総時間を管理できるように分散させます。
# MAGIC 
# MAGIC ハイパーパラメータについては、隠れユニットの数と、モデルのディープニューラルネットワーク部分のドロップアウト率を調整します。 また、学習のエポック数やバッチサイズを調整することもできますが、今回は固定値にしています。

# COMMAND ----------

# DBTITLE 1,ハイパーパラメータ探索空間の定義
search_space = {
  'hidden_layers': hp.quniform('hidden_layers', 1, 5, 1)  # determines number of hidden layers
  ,'hidden_layer_nodes_initial_count': hp.quniform('hidden_layer_nodes_initial', 50, 201, 10)  # determines number of nodes in first hidden layer
  ,'hidden_layer_nodes_count_decline_rate': hp.quniform('hidden_layer_nodes_count_decline_rate', 0.0, 0.51, 0.05) # determines how number of nodes decline in layers below first hidden layer
  ,'dropout_rate': hp.quniform('dropout_rate', 0.0, 0.51, 0.05)
  ,'epochs': hp.quniform('epochs', 3, 4, 1) # fixed value for now
  ,'batch_size': hp.quniform('batch_size', 128, 129, 1) # fixed value for now
  }

# COMMAND ----------

# DBTITLE 1,ハイパーパラメータ探索の実行
argmin = fmin(
  fn=train_and_evaluate_model,
  space=search_space,
  algo=tpe.suggest,
  max_evals=100,
  trials=SparkTrials(parallelism=sc.defaultParallelism) # set to the number of executors for CPU-based clusters OR number of workers for GPU-based clusters
  )

# COMMAND ----------

# DBTITLE 1,最適化されたハイパーパラメータの表示
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md ## Step 4: モデルの評価
# MAGIC 
# MAGIC 最適化されたパラメータに基づいて、モデルの最終バージョンをトレーニングし、それに関連するメトリクスを調べることができます。

# COMMAND ----------

# DBTITLE 1,最適化されたトレーニングモデル
hparams = space_eval(search_space, argmin)

model = get_model(
    hparams['hidden_layers'], 
    hparams['hidden_layer_nodes_initial_count'], 
    hparams['hidden_layer_nodes_count_decline_rate'], 
    hparams['dropout_rate']
    )
model = tf.estimator.add_metrics(model, map_custom_metric)

train_spec, eval_spec = get_data_specs(hparams['epochs'],hparams['batch_size'])

results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

# COMMAND ----------

# DBTITLE 1,バリデーション結果の確認
results[0]

# COMMAND ----------

# MAGIC %md ハイパーパラメータのチューニング時にモデルが見ていないテストデータを使用することで、モデルの性能をよりよく評価することができます。 テストデータもPetastormに保存されていますが、評価のためにデータを再編成するための関数にアクセスする必要があります。 さらに、データを評価するデータステップ数を明示的に定義する必要があります（さもなければ、評価ステップは無限に実行されます）。

# COMMAND ----------

# DBTITLE 1,テストデータで評価する
# Borrowed from get_data_specs() (defined above)
# ---------------------------------------------------------
# define functions to transform data into req'ed format
def get_input_fn(dataset_context_manager):

  def _to_tuple(row): # re-structure a row as ({features}, label)
    features = {}
    for col in cat_features + num_features:
      features[col] = getattr(row, col)
    return features, getattr(row, label_col)

  def fn(): # called by estimator to perform row structure conversion
    return dataset_context_manager.__enter__().map(_to_tuple)

  return fn
# ---------------------------------------------------------

# define batch size and number of steps
batch_size = 128
steps = int(test_pstorm.dataset_size/batch_size)

# retrieve test data
test_ds = test_pstorm.make_tf_dataset(batch_size=batch_size)

# evaulate against test data
results = model.evaluate(get_input_fn(test_ds), steps=steps)

# COMMAND ----------

# DBTITLE 1,テスト結果の確認
# show results
results

# COMMAND ----------

# MAGIC %md 今回開発したモデルは、テスト用のホールドアウトでも同じような結果が得られるようです。 このモデルを本番テスト用のアプリケーション・インフラに移行することに自信を持つべきでしょう。

# COMMAND ----------

# MAGIC %md ## Step 5: モデルのデプロイ
# MAGIC 
# MAGIC トレーニングを受けて評価されたモデルを、アプリケーション・インフラストラクチャに移行する必要があります。 これを行うには、デプロイを可能にする方法でモデルを永続化する必要があります。このためにはMLflowを利用するが、その前にTensorflowの組み込み機能を使ってモデルを永続化する必要があるだろう。これにより、後でMLflowがピクルス化されたモデルをピックアップするのが容易になります。

# COMMAND ----------

# DBTITLE 1,Tensorflowモデルの一時的なエクスポート
# get features
wide_features = get_wide_features()
deep_features = get_deep_features()

# use features to generate an input specification
feature_spec = tf.feature_column.make_parse_example_spec(
    wide_features + deep_features
    )

# make function to apply specification to incoming data
fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    feature_spec
    )

# export the model
saved_model_path = model.export_saved_model(
    export_dir_base='/dbfs/tmp/exported_model',
    serving_input_receiver_fn=fn
    ).decode("utf-8")

# COMMAND ----------

# MAGIC %md Tensorflowモデルは、特定の形式のデータを期待し、各予測でいくつかの値を返します。 このラッパーは、ピクルス化されたTensorflowモデルをピックアップし、入力データをそのモデルが理解できるフォーマットに変換し、各予測で正のクラス確率を返す。

# COMMAND ----------

# DBTITLE 1,モデルのラッパーの定義
# custom wrapper to align model with mlflow
class Recommender(mlflow.pyfunc.PythonModel):
  
  # The code snippet in this cell can be reused on new datasets with some modifications to match the feature types of your dataset. 
  # see docs: https://www.tensorflow.org/tutorials/load_data/tfrecord
  def _convert_inputs(self, inputs_pd):

    proto_tensors = []

    # for each row in the pandas dataframe
    for i in range(len(inputs_pd)):

      # translate field values into features
      feature = dict()
      for field in inputs_pd:
        if field not in ['id','aisle_id','department_id','user_id','product_id']:
          feature[field] = tf.train.Feature(float_list=tf.train.FloatList(value=[inputs_pd[field][i]]))
        else: 
          feature[field] = tf.train.Feature(int64_list=tf.train.Int64List(value=[inputs_pd[field][i]]))

      # convert rows into expected format
      proto = tf.train.Example(features=tf.train.Features(feature=feature))
      proto_string = proto.SerializeToString()
      proto_tensors.append(tf.constant([proto_string]))

    return proto_tensors
  
  # load saved model upon initialization
  def load_context(self, context):
    self.model = mlflow.tensorflow.load_model(context.artifacts['model'])  # retrieve the unaltered tensorflow model (persisted as an artifact)

  # logic to return scores
  def predict(self, context, model_input):
    # convert inputs into required format
    proto = self._convert_inputs(model_input)
    # score inputs
    results_list = []
    for p in proto:
      results_list.append(self.model(p))
    # retrieve positive class probability as score 
    ret = [item['probabilities'][0, 1].numpy() for item in results_list]
    return ret

# COMMAND ----------

# MAGIC %md 私たちのモデルは、大量の機能が渡されることを想定しています。 データ構造の要件を明確にするために、[モデルと一緒に永続化するサンプルデータセット](https://www.mlflow.org/docs/latest/models.html#input-example)を作成します。

# COMMAND ----------

# DBTITLE 1,サンプル入力データセットの構築
# use user 123 as sample user
user_id = spark.createDataFrame([(123,)], ['user_id'])

# get features for user and small number of producs
sample_pd = (
  user_id
    .join(user_features, on='user_id') # get user features
    .crossJoin(product_features.limit(5)) # get product features (for 5 products)
  ).toPandas()

# show sample
sample_pd

# COMMAND ----------

# MAGIC %md これでようやく、モデルをMLflowに永続化することができます。後のステップで役立つように、わかりやすい名前を使ってモデルを登録し、このモデルのライブラリの依存関係についての情報を含めます。

# COMMAND ----------

# DBTITLE 1,モデル名のID
model_name='recommender'

# COMMAND ----------

# DBTITLE 1,mlflowでモデルを永続化させる
# libraries for these models
conda_env = {
    'channels': ['defaults'],
    'dependencies': [
      f'python={platform.python_version()}',
      'pip',
      {
        'pip': [
          'mlflow',
          #f'tensorflow-gpu=={tf.__version__}',  # gpu-version
          f'tensorflow-cpu=={tf.__version__}',   # cpu-version
          'tensorflow-estimator',
        ],
      },
    ],
    'name': 'recommender_env'
}

# create an experiment run under which to store these models
with mlflow.start_run(run_name=model_name) as run:
  
  # log the tensorflow model to mlflow
  tf_meta_graph_tags = [tag_constants.SERVING]
  tf_signature_def_key='predict'
  
  # persist the original tensorflow model
  mlflow.tensorflow.log_model(
    tf_saved_model_dir=saved_model_path, 
    tf_meta_graph_tags=tf_meta_graph_tags, 
    tf_signature_def_key=tf_signature_def_key,
    artifact_path='model',
    conda_env=conda_env
    )
  
  # retrieve artifact path for the tensorflow model just logged
  artifacts = {
    # Replace the value with the actual model you logged
    'model': mlflow.get_artifact_uri() + '/model'
  }

  # record the model with the custom wrapper with the tensorflow model as its artifact
  mlflow_pyfunc_model_path = 'recommender_mlflow_pyfunc'
  mlflow.pyfunc.log_model(
    artifact_path=mlflow_pyfunc_model_path, 
    python_model=Recommender(), 
    artifacts=artifacts,
    conda_env=conda_env, 
    input_example=sample_pd)

  # register last deployed model with mlflow model registry
  mv = mlflow.register_model(
      'runs:/{0}/{1}'.format(run.info.run_id, mlflow_pyfunc_model_path),
      model_name
      )
  
  # record model version for next step
  model_version = mv.version

# COMMAND ----------

# MAGIC %md 最後のセルのステップの最後に、MLflow registryにモデルを登録します。 [MLflow registry](https://www.mlflow.org/docs/latest/model-registry.html)は、モデルを初期状態からステージング、プロダクション、そしてアーカイブへと昇格させるためのサポートを提供します。組織は、モデルを評価してから現在の本番インスタンスとして指定できるようなワークフローを構築することができます。ここでは、公開したばかりのモデルを、テストや依存するシステムとの調整を行わずに、本番環境に直接プッシュすることにしますが、これはデモ以外では推奨できません。

# COMMAND ----------

# DBTITLE 1,Promote ModelをProduction Statusにプッシュする
client = mlflow.tracking.MlflowClient()

# archive any production versions of the model from prior runs
for mv in client.search_model_versions("name='{0}'".format(model_name)):
  
    # if model with this name is marked production
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='archived'
        )
      
# transition newly deployed model to production stage
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )     

# COMMAND ----------

# MAGIC %md MLflowのレジストリには、ユーザーと製品の組み合わせをスコアリングするためのモデルが用意されています。 アプリケーションのページが、検索語やカテゴリの選択に関連した製品のサブセットを提示し、それらの製品の特徴とユーザーの特徴をモデルに送信し、ページ上の製品を注文するために使用されるスコアを受け取る、というシナリオが考えられます。
# MAGIC 
# MAGIC このシナリオでは、モデルの提供は、[Azure ML](https://www.mlflow.org/docs/latest/python_api/mlflow.azureml.html)、[AWS Sagemaker](https://www.mlflow.org/docs/latest/python_api/mlflow.sagemaker.html)、または[Model Serving](https://docs.databricks.com/applications/mlflow/model-serving.html)機能を備えたDatabricks自身にホストされたマイクロサービスを通じて行われるかもしれません。基本的にMLflowはモデルを[Docker image](https://www.mlflow.org/docs/latest/cli.html#mlflow-models-build-docker)にデプロイし、あらかじめ定義されたRESTエンドポイントで公開しているため、他にも様々なデプロイ方法が考えられます。 そう考えると、[Kubernetes](https://www.mlflow.org/docs/latest/projects.html#run-an-mlflow-project-on-kubernetes-experimental)のような技術が、実行可能なデプロイメントパスとして浮かび上がってきます。
# MAGIC 
# MAGIC ここでは、DatabricksのModel Servingを利用してみましょう。 Serving UIにアクセスするには、画面左上のドロップダウンをクリックして、Databricks UIのMachine Learningインターフェースに切り替えます。Databricks UIの左側にあるModelsアイコン<img src='https://brysmiwasb.blob.core.windows.net/demos/images/widedeep_models_icon.PNG' width=50>をクリックすると、先ほどの手順で登録したモデルを見つけることができます。 そのモデルをクリックすると、2つのタブが表示されます。Details」と「Serving」です。 Serving」タブをクリックすると、「<img src='https://brysmiwasb.blob.core.windows.net/demos/images/widedeep_enableserving_button.PNG'>」ボタンを選択して、モデルをホストする小さなシングルノードのクラスタを起動することができます。
# MAGIC 
# MAGIC このボタンをクリックすると、呼び出したいモデルの製品版を選択することができます。 選択したモデルのバージョンに応じて、モデルのURLが表示されることに注意してください。 次に、以下のコードを使用して、このモデルに関連するREST APIにデータを送信することができます。
# MAGIC 
# MAGIC このモデルを実行しているシングルノード・クラスターは、リクエストを受け付ける前に**Pending**状態から**Running**状態に移行する必要があることに注意してください。 また、このクラスタは、［Serving］タブに戻り、ステータスの横にある［Stop］を選択しない限り、無期限に実行されます。
# MAGIC 
# MAGIC 最後に、Databricks Model Servingが提供するREST APIは、[Databricks Personal Access Token](https://docs.databricks.com/dev-tools/api/latest/authentication.html)を使用して保護されています。このようなトークンは、以下のようにコードに埋め込まないことが推奨されるベストプラクティスですが、透明性を確保するためにこの推奨に違反しています。

# COMMAND ----------

# DBTITLE 1,ユーザーと製品の組み合わせを検索してスコアにする
user_id = 123 # user 123
aisle_id = 32 # packaged produce department

# retrieve user features
user_features = (
  spark
    .table('instacart.user_features')
    .filter(f.expr('user_id={0}'.format(user_id)))
  )

# retrieve features for product in department
product_features = (
  spark
    .table('instacart.product_features')
    .filter('aisle_id={0}'.format(aisle_id))
  )

# combine for scoring
user_products = (
  user_features
    .crossJoin(product_features)
    .toPandas()
    )

# show sample of feature set
user_products.head(5)

# COMMAND ----------

# DBTITLE 1,スコアの検索
personal_access_token='dapi27074f9636590d32146f2e3d59afc2aa'
databricks_instance='adb-2704554918254528.8.azuredatabricks.net' 
model_name = 'recommender'

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  # prepare data for transmission
  data_json = dataset.to_dict(orient='records') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  #data_json = dataset.to_dict(orient='records')
  
  # REST API settings
  url = 'https://{0}/model/{1}/Production/invocations'.format(databricks_instance,model_name)
  headers = {'Authorization': 'Bearer {0}'.format(personal_access_token)}
  
  # send data to REST API for scoring
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    
  # return json output
  return response.json()

# call REST API for scoring
score_model(user_products)

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
