# Databricks notebook source
# MAGIC %md 
# MAGIC # Databricks上でのDeep Learning:
# MAGIC #####Keras + Hyperopt + MLflowによるEnd-to-Endサンプル
# MAGIC 
# MAGIC Original Notebook: [getting-started-keras](https://docs.databricks.com/_static/notebooks/getting-started/get-started-keras-dbr7ml.html)
# MAGIC 
# MAGIC このチュートリアルでは、小さなデータセットを使って、TensorFlow Keras、Hyperopt、MLflowを使ってDatabricksで深層学習モデルを開発する方法を紹介します。
# MAGIC 
# MAGIC 以下のステップで説明していきます:
# MAGIC - データの読み込み、および、前処理(preprocessing)
# MAGIC - Part 1. TensorFlow Kerasでニューラルネットワークモデルを作成し、インラインのTensorBoardでトレーニングを表示する
# MAGIC - Part 2. HyperoptおよびMLflowを用いたハイパーパラメータチューニングの自動化と、オートロギングによる結果の保存
# MAGIC - Part 3. 最適なハイパーパラメータのセットを使用して、最終的なモデルを構築する 
# MAGIC - Part 4. MLflowにモデルを登録し、そのモデルを使って予測を行う
# MAGIC 
# MAGIC ### セットアップ
# MAGIC - Databricks Runtime ML 7.0以上を使用しています。このノートブックでは、ニューラルネットワークの学習結果の表示にTensorBoardを使用しています。使用しているDatabricks Runtimeのバージョンによって、TensorBoardを起動する方法が異なります。

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

# COMMAND ----------

# MAGIC %md ## データの読み込み、および、前処理(preprocessing)
# MAGIC この例では`scikit-learn`のCalifornia Housingデータセットを使用していま

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

cal_housing = fetch_california_housing()

# サンプルデータを80/20でトレーニングデータ、テストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2)

# COMMAND ----------

# MAGIC %md ### 特徴量のスケーリング
# MAGIC 
# MAGIC ニューラルネットワークを扱う際には，特徴量のスケーリングが重要になります．このノートブックでは，`scikit-learn`関数の`StandardScaler`を使用します．

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# T_train, y_trainデータの確認
import pandas as pd
display( 
  pd.concat(
    [pd.DataFrame(X_train, columns=cal_housing.feature_names), pd.DataFrame(y_train, columns=["label"])], axis=1)
)

# COMMAND ----------

# MAGIC %md ## Part 1. モデルの作成、TensorBoradによる視覚化

# COMMAND ----------

# MAGIC %md ### ニューラルネットワークの作成

# COMMAND ----------

def create_model():
  model = Sequential()
  model.add(Dense(20, input_dim=8, activation="relu"))
  model.add(Dense(20, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### モデルのコンパイル

# COMMAND ----------

# NNのモデル構成(インスタンス化)
model = create_model()

# ハイパーパラメータをセットしてコンパイル
model.compile(loss="mse",  # <= Loss関数としてMSE(Mean Squared Error)を使用
              optimizer="Adam", # <= アルゴリズム"Adam"を使用して最適化
              metrics=["mse"]) # <= メトリックとしてMSEを用いる

# COMMAND ----------

# MAGIC %md ### コールバックの作成

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# <username>　を適宜置換してください!!
experiment_log_dir = "/dbfs/tmp/kitamura123/tb"
checkpoint_path = "/dbfs/tmp/kitamura123/keras_checkpoint_weights.ckpt"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)

history = model.fit(X_train, y_train, validation_split=.2, epochs=35, callbacks=[tensorboard_callback, model_checkpoint, early_stopping])

# COMMAND ----------

# MAGIC %md ### TensorBoardコマンド( Databrciks Runtime ML 7.2以上が必要)
# MAGIC 
# MAGIC この方法でTensorBoardを起動すると、ノートブックをクラスターから切り離すまで、TensorBoardが実行され続けます。
# MAGIC 
# MAGIC 注：実行の間にTensorBoardをクリアするには、次のコマンドを使用します: `dbutils.fs.rm(experiment_log_dir.replace("/dbfs",""), recurse=True)`

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# MAGIC %md ### モデルの評価

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md ## Part 2. HyperoptとMLflowを用いたハイパーパラメータチューニング
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt)は、ハイパーパラメータチューニングのためのPythonライブラリです。Databricks Runtime for MLには、自動化されたMLflowのトラッキングを含む、最適化され強化されたHyperoptのバージョンが含まれています。Hyperoptの使用に関する詳細は、[Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)を参照してください。

# COMMAND ----------

# MAGIC %md ### ニューラルネットワークの作成 (hiddenレイヤ数を引数にした関数)

# COMMAND ----------

def create_model(n):
  model = Sequential()
  model.add(Dense(int(n["dense_l1"]), input_dim=8, activation="relu"))
  model.add(Dense(int(n["dense_l2"]), activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### Hyperoptの目的関数(objective function)の作成

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

def runNN(n):
  # Import tensorflow 
  import tensorflow as tf
  
  # MLflowの自動トラッキングをONにする
  mlflow.tensorflow.autolog()
  
  model = create_model(n)

  # Optimizerの設定
  optimizer_call = getattr(tf.keras.optimizers, n["optimizer"])
  optimizer = optimizer_call(learning_rate=n["learning_rate"])
 
  # モデルのコンパイル
  model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

  history = model.fit(X_train, y_train, validation_split=.2, epochs=10, verbose=2)

  # モデルの評価
  score = model.evaluate(X_test, y_test, verbose=0)
  obj_metric = score[0]  
  return {"loss": obj_metric, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md ### Hyperoptのsearch空間を定義する(探索するパラメータの範囲)

# COMMAND ----------

space = {
  "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
  "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
  "learning_rate": hp.loguniform("learning_rate", -5, 0),
  "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
 }

# COMMAND ----------

# MAGIC %md ### `SparkTrials` オブジェクトの作成
# MAGIC 
# MAGIC SparkTrials`オブジェクトは`fmin()`にチューニングジョブをSparkクラスタに分散するように指示します。SparkTrials`オブジェクトを作成する際には，`parallelism`引数を使って，同時評価する試行の最大数を設定することができます．デフォルトの設定は、利用可能なSparkノード数になります。 
# MAGIC 
# MAGIC 数値が大きいほど、より多くのハイパーパラメータ設定のテストをスケールアウトすることができます。Hyperoptは過去の結果に基づいて新しいテストを提案するので、並列性と適応性の間にはトレードオフがあります。固定の`max_evals`では、並列性が高いほど計算が速くなりますが、並列性が低いほど、各反復がより多くの過去の結果にアクセスできるため、より良い結果が得られる可能性があります。

# COMMAND ----------

# `parallelism`を指定しない場合は、Sparkのexecutor数がデフォルトで使用される
spark_trials = SparkTrials()

# COMMAND ----------

# MAGIC %md ### ハイパーパラメータチューニングを実行する
# MAGIC 
# MAGIC 結果をMLflowに保存するには、MLflowの実行スコープ(`with`句内)で`fmin()`を呼びます。MLflowは、各ランのパラメータとパフォーマンスメトリクスを追跡します。  
# MAGIC 
# MAGIC 以下のセルを実行した後、その結果をMLflowで確認することができます。右上の **Experiment** をクリックして、Experiment Runs サイドバーを表示します。右端の「Experiment Runs」の隣にあるアイコンをクリックすると、「MLflow Runs Table」が表示されます。
# MAGIC 
# MAGIC MLflowを使ったモデル学習の解析については、([AWS](https://docs.databricks.com/applications/mlflow/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/))を参照してください。

# COMMAND ----------

with mlflow.start_run():
  best_hyperparam = fmin(fn=runNN, 
                         space=space, 
                         algo=tpe.suggest, 
                         max_evals=30, 
                         trials=spark_trials)

# COMMAND ----------

# MAGIC %md ## Part 3. 最後に最適なハイパーパラメータを使ってモデルを構築する

# COMMAND ----------

import hyperopt

print(hyperopt.space_eval(space, best_hyperparam))

# COMMAND ----------

first_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l1"]
second_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l2"]
learning_rate = hyperopt.space_eval(space, best_hyperparam)["learning_rate"]
optimizer = hyperopt.space_eval(space, best_hyperparam)["optimizer"]

# COMMAND ----------

# Get optimizer and update with learning_rate value
optimizer_call = getattr(tf.keras.optimizers, optimizer)
optimizer = optimizer_call(learning_rate=learning_rate)

# COMMAND ----------

def create_new_model():
  model = Sequential()
  model.add(Dense(first_layer, input_dim=8, activation="relu"))
  model.add(Dense(second_layer, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

new_model = create_new_model()
  
new_model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC `autolog()`がアクティブな場合、MLflowは自動的にrunを終了しません。新しいランを開始してオートログする前に、Cmd28で開始したrunを終了する必要があります。 
# MAGIC 詳しくは、https://www.mlflow.org/docs/latest/tracking.html#automatic-logging。

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import matplotlib.pyplot as plt

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  
  history = new_model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])
  
  # Save the run information to register the model later
  kerasURI = run.info.artifact_uri
  
  # Evaluate model on test dataset and log result
  mlflow.log_param("eval_result", new_model.evaluate(X_test, y_test)[0])
  
  # Plot predicted vs known values for a quick visual check of the model and log the plot as an artifact
  keras_pred = new_model.predict(X_test)
  plt.plot(y_test, keras_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("kplot.png")
  mlflow.log_artifact("kplot.png") 

# COMMAND ----------

# MAGIC %md ## Part 4. MLflowにモデルを登録し、そのモデルを使って予測を行う
# MAGIC 
# MAGIC モデルレジストリの詳細については、([AWS](https://docs.databricks.com/applications/mlflow/model-registry.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-registry))をご覧ください。

# COMMAND ----------

import time

model_name = "cal_housing_keras"
model_uri = kerasURI+"/model"
new_model_version = mlflow.register_model(model_uri, model_name)

# Registering the model takes a few seconds, so add a delay before continuing with the next cell
time.sleep(5)

# COMMAND ----------

# MAGIC %md ### 推論のためにモデルを読み込み、予測を行う

# COMMAND ----------

# MLflowレジストリからモデルを読み込む(pull)
# (モデル名とversionを指定でロード可能)
keras_model = mlflow.keras.load_model(f"models:/{model_name}/{new_model_version.version}")

keras_pred = keras_model.predict(X_test)
keras_pred

# COMMAND ----------

# MAGIC %md ## Clean up
# MAGIC TensorBoardを終了するには:
# MAGIC - Databricks Runtime 7.1 MLもしくはそれ以前のversion => 下のセルのコメントを外して実行する
# MAGIC - Databricks Runtime 7.1 MLもしくはそれ以降のversion => クラスタからnotebookをデタッチする

# COMMAND ----------

#dbutils.tensorboard.stop()
