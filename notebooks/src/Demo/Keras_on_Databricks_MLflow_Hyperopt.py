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
# MAGIC - データの読み込み、および、前処理 --- (preprocessing, この例ではシングルコンピュート。ただし、Spakr分散処理も可)
# MAGIC - Part 1. Kerasの基礎 --- (シングルコンピュート)
# MAGIC - Part 2. MLflowの基礎 - Kerasの実験結果をトラックする --- (シングルコンピュート)
# MAGIC - Part 3. HyperoptおよびMLflowを用いたハイパーパラメータチューニングの自動化 --- (Spark分散処理)
# MAGIC - Part 4. 最適なハイパーパラメータのセットを使用して、最終的なモデルを構築する --- (シングルコンピュート)
# MAGIC - Part 5. MLflowにモデルを登録し、そのモデルを使って予測を行う --- (Spark分散処理 or シングルコンピュート)
# MAGIC 
# MAGIC ### セットアップ
# MAGIC - Databricks Runtime ML 7.0以上を使用しています。このノートブックでは、ニューラルネットワークの学習結果の表示にTensorBoardを使用しています。使用しているDatabricks Runtimeのバージョンによって、TensorBoardを起動する方法が異なります。

# COMMAND ----------

# DBTITLE 1,ユニークなパス名を設定(ユーザー間の衝突回避)
import re

# Username を取得。
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Usernameをファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '_', username_raw).lower()

print(f'>>> username => {username}')

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
import mlflow.tensorflow

# COMMAND ----------

# MAGIC %md ## Part0. データの読み込み、および、前処理(preprocessing)
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

# X_train, y_trainデータの確認
import pandas as pd
display( 
  pd.concat(
    [pd.DataFrame(X_train, columns=cal_housing.feature_names), pd.DataFrame(y_train, columns=["label"])], axis=1)
)

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

# MAGIC %md ## Part 1. モデルの作成、TensorBoradによる視覚化

# COMMAND ----------

# MAGIC %md ### ニューラルネットワークの作成

# COMMAND ----------

def create_model():
  model = Sequential() # NNレイヤのインスタンスを用意
  model.add(Dense(20, input_dim=8, activation="relu"))  # <= Inputレイヤ 
  model.add(Dense(20, activation="relu"))  # <= 中間レイヤ
  model.add(Dense(1, activation="linear")) # <= Outputレイヤ
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

# MAGIC %md ### モデル学習(TensorBoard向けのコールバック作成を含む)

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 中間ファイルの保存先を指定
experiment_log_dir = f"/dbfs/tmp/{username}/tb"
checkpoint_path = f"/dbfs/tmp/{username}/keras_checkpoint_weights.ckpt"

# 過去の中間ファイル残骸を削除
dbutils.fs.rm(experiment_log_dir, True)
dbutils.fs.rm(checkpoint_path, True)


# fit関数に入れるパラメータ(関数)を設定
# (Tensorboardを使用するためのコード)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=experiment_log_dir)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor="loss", mode="min", patience=3)


# 学習(fit)の実行
history = model.fit(
  X_train, y_train, validation_split=.2, epochs=35,
  callbacks=[tensorboard_callback, model_checkpoint, early_stopping]
)

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

# MAGIC %md
# MAGIC 
# MAGIC ## Part2: MLFlowを用いて、学習をトラックする
# MAGIC 
# MAGIC 上記で実施したモデル学習をMLflowトラックするためのコードに書き換える(ほぼ同じコード)

# COMMAND ----------

import mlflow
mlflow.autolog() # <= MLflowのAutoトラッキングを有効化する


# 学習結果のプロットファイルを出力するための関数を用意(これをMLflowでトラックする際に使用)
import matplotlib.pyplot as plt
def viewModelLoss(history):
  plt.clf()
  plt.semilogy(history.history["loss"], label="train_loss")
  plt.title("Model Loss")
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.legend()
  return plt



# このwith節内のコードをMLflowが自動でトラックする
with mlflow.start_run():
  
  # NNレイヤのインスタンスを用意
  model = Sequential() 
  model.add(Dense(20, input_dim=8, activation="relu"))  # <= Inputレイヤ 
  model.add(Dense(20, activation="relu"))  # <= 中間レイヤ
  model.add(Dense(1, activation="linear")) # <= Outputレイヤ
  
  # ハイパーパラメータをセットしてコンパイル
  model.compile(loss="mse",  # <= Loss関数としてMSE(Mean Squared Error)を使用
              optimizer="Adam", # <= アルゴリズム"Adam"を使用して最適化
              metrics=["mse"]) # <= メトリックとしてMSEを用いる
  
  # 学習を実施(fit)
  history = model.fit(X_train, y_train, validation_split=.2, epochs=35)
  
  # 評価
  model.evaluate(X_test, y_test)
  
  
  # mlflowのタグをつける
  mlflow.log_param('Model_Type', 'ABC123')
  
  # mlflowで画像も含め、あらゆるデータを記録しておく
  fig = viewModelLoss(history)
  fig.savefig("train-validation-loss.png")
  mlflow.log_artifact("train-validation-loss.png")
  

# COMMAND ----------

# MAGIC %md ## Part 3. MLflow + Hyperoptを用いたハイパーパラメータチューニング
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt)は、ハイパーパラメータチューニングのためのPythonライブラリです。Databricks Runtime for MLには、自動化されたMLflowのトラッキングを含む、最適化され強化されたHyperoptのバージョンが含まれています。Hyperoptの使用に関する詳細は、[Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin)を参照してください。

# COMMAND ----------

# MAGIC %md ### ニューラルネットワークの作成 (Input, Hiddenレイヤ数を引数にした関数)

# COMMAND ----------

# Input, Hiddenレイヤ数を引数に、モデルのインスタンスを作成する関数
def create_model(n):
  model = Sequential()
  model.add(Dense(int(n["dense_l1"]), input_dim=8, activation="relu"))
  model.add(Dense(int(n["dense_l2"]), activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model

# COMMAND ----------

# MAGIC %md ### Hyperoptの目的関数(objective function)の作成
# MAGIC 
# MAGIC HyperOptは与えられた目的関数を最小にするようにハイパーパラメータを探索する。基本的に、この関数の中でモデル学習を実施して、モデルの評価値(小さい方がベターという値、例えばMSEなど)を戻り値として返すような目的関数を実装する。

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

# Hyperoptに渡す目的関数:
# n: Hyperoptによって渡されるハイパーパラメータが入る。具体的には下セルの`space`で定義されている内容。
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
spark_trials = SparkTrials(parallelism=24)

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

# MAGIC %md ## Part 4. 最後に最適なハイパーパラメータを使ってモデルを構築する

# COMMAND ----------

import hyperopt

# 最適化の結果パラメータの確認
print(hyperopt.space_eval(space, best_hyperparam))

# 最適化の結果パラメータを取り出す
first_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l1"]
second_layer = hyperopt.space_eval(space, best_hyperparam)["dense_l2"]
learning_rate = hyperopt.space_eval(space, best_hyperparam)["learning_rate"]
optimizer = hyperopt.space_eval(space, best_hyperparam)["optimizer"]


# COMMAND ----------

# 再度上記のパラメータで学習を実施する

# 最適なパラメータでOptimizerを設定する
optimizer_call = getattr(tf.keras.optimizers, optimizer)
optimizer = optimizer_call(learning_rate=learning_rate)


# モデルのインスタンス化 (最適なパラメータ)
def create_new_model():
  model = Sequential()
  model.add(Dense(first_layer, input_dim=8, activation="relu"))
  model.add(Dense(second_layer, activation="relu"))
  model.add(Dense(1, activation="linear"))
  return model


# 上記の関数を実行する
new_model = create_new_model()
new_model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC `autolog()`がアクティブな場合、MLflowは自動的にrunを終了しません。新しいランを開始してオートログする前に、Cmd28で開始したrunを終了する必要があります。 
# MAGIC 詳しくは、https://www.mlflow.org/docs/latest/tracking.html#automatic-logging 。

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

import matplotlib.pyplot as plt

# mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  
  history = new_model.fit(X_train, y_train, epochs=35, callbacks=[early_stopping])
  
  # Save the run information to register the model later
  kerasURI = run.info.artifact_uri
  
  # Evaluate model on test dataset and log result
  #mlflow.log_param("eval_result", new_model.evaluate(X_test, y_test)[0])
  
  # プロットデータもMLflowに記録しておく
  keras_pred = new_model.predict(X_test)
  plt.plot(y_test, keras_pred, "o", markersize=2)
  plt.xlabel("observed value")
  plt.ylabel("predicted value")
  plt.savefig("kplot.png")
  mlflow.log_artifact("kplot.png") 

# COMMAND ----------

# MAGIC %md ### MLflowのモデルトラッキングで実験結果を比較する

# COMMAND ----------

# MAGIC %md ## Part 5. MLflowにモデルを登録
# MAGIC 
# MAGIC 上記で作成できた最適なモデルをMLflowのモデルレジストリに登録し、version管理をしていきましょう。
# MAGIC 
# MAGIC <div><img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/model-registry.png" style="height: 400px; margin: 20px"/></div>
# MAGIC 
# MAGIC  See <a href="https://mlflow.org/docs/latest/registry.html" target="_blank">the MLflow docs</a> for more details on the model registry.
# MAGIC 
# MAGIC モデルレジストリの詳細については、([AWS](https://docs.databricks.com/applications/mlflow/model-registry.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/model-registry))をご覧ください。

# COMMAND ----------

# DBTITLE 1,モデルレジストリへの登録はコードからもでも可能
# import time

# model_name = "cal_housing_keras"
# model_uri = kerasURI+"/model"
# new_model_version = mlflow.register_model(model_uri, model_name)

# # Registering the model takes a few seconds, so add a delay before continuing with the next cell
# time.sleep(5)

# COMMAND ----------

# MAGIC %md ## Part6. デプロイの推論(MLflowからモデルをロードする)
# MAGIC 
# MAGIC Databricks上でのモデルのデプロイは以下の3通りに分けられます。
# MAGIC 
# MAGIC 1. バッチ処理: Databricks上のnotebookでDataframeを入力し、スコアリングするコードを定期実行する
# MAGIC 1. ストリーミング処理: Databricks上のnotebookでストリーミングDataframeを入力し、スコアリングを逐次実行する
# MAGIC 1. REST Serving: REST Server上にモデルをデプロイし、HTTPリクエストでスコアリングデータを読み込み、レスポンスで推定結果を返す
# MAGIC 
# MAGIC Databricks上ではバッチ処理、ストリーミング処理がDataframe的に同等に扱えるため、上記のバッチ処理、ストリーミング処理はほぼ同じデプロイ方法になります。
# MAGIC Rest Servingについては、MLflowのレジストリUIからデプロイ可能です。
# MAGIC 
# MAGIC ここでは、バッチ処理、ストリーミング処理でデプロイする方法を見ていきます。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC MLflowからモデルをロードする場合、以下の2通りがあります。
# MAGIC 
# MAGIC 1. **Run(学習の結果)**のIDを指定してロードする
# MAGIC 1. モデルレジストリから、**モデル名、version(もしくは、staging, production)**を指定してロードする
# MAGIC 
# MAGIC さらに、ロードしたモデルは、以下の2通りの種類でロードできます。
# MAGIC 
# MAGIC 1. **Spark**のDataframeを入力できるPythonの関数としてロード (**Spark分散処理、かつ、ストリーミングにも対応**)
# MAGIC 1. **Pandas**のDataframeを入力できるPythonの関数としてロード (**シングルコンピュート**)

# COMMAND ----------

# MAGIC %md
# MAGIC ### A. Run IDからSpark Dataframeのpython関数としてデプロイする

# COMMAND ----------

import mlflow

# 実際のRun IDで置き換えてください!
logged_model = 'runs:/1237b25ceb9d4cbaa7c475923672d804/model'

# モデルをロードする
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
loaded_model

# COMMAND ----------

# 推定を実施する(スコアリングを実施する)対象のデータを読み込む

df = spark.createDataFrame( pd.DataFrame(X_train, columns=cal_housing.feature_names) )
display(df)

# COMMAND ----------

# モデルを適用して推定する(スコアリング)
pred_df = df.withColumn('pred', loaded_model(*df.columns))
display(pred_df)

# COMMAND ----------

# MAGIC %md ### B. Run IDからPandas Dataframeのpython関数としてデプロイする

# COMMAND ----------

import mlflow

# 実際のRun IDで置き換えてください!
logged_model = 'runs:/1237b25ceb9d4cbaa7c475923672d804/model'

pd_model = mlflow.pyfunc.load_model(logged_model)
pd_model

# COMMAND ----------

# 推定を実施する(スコアリングを実施する)対象のデータを読み込む
# (先ほど読み込んだSpark DataframeをPandas Dataframeに変換する)

pd_df = pd.DataFrame(X_train, columns=cal_housing.feature_names)
pd_df

# COMMAND ----------

# モデルを適用して推定する(スコアリング)
import pandas as pd
pred_array = pd_model.predict(pd_df)

# 結果をDataframeにまとめて、表示
pd_df['pred'] = pred_array
pd_df

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### C. モデルレジストリから、モデル名、version(もしくは、staging, production)を指定してロードする

# COMMAND ----------

import mlflow.pyfunc

model_name = "レジストリ上のモデルの名前"
model_version = 1
# model_version = 'production' ## <= このようにproduction/stagingも指定可能

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

# COMMAND ----------

# MAGIC %md ## (参考) デプロイの推論(MLflowのRESTモデルサービングを使用する)
# MAGIC 
# MAGIC MLflowのモデルレジストリのUIからRESTサービスとしてモデルをデプロイすることが可能です。
# MAGIC RESTサービスのモデルServingを使用すると、以下のようにCurlからHTPリクエストによってモデルのスコアリング(推定)が可能です。

# COMMAND ----------

# 入力データを用意(HTTPリクエストのBodyで送信される)
data_json = json.dumps( pd_df[0:3].to_dict(orient="split") )

# Access Tokenを用意(Databricksのユーザー設定から発行できます)
token = 'xxxxxxxxxxxxxxxxxxxxxxx'

# モデルサービングのURL (MLflowのモデルサービングUIから参照可能です)
url = 'https://demo.cloud.databricks.com/model/ktmr_keras_demo/1/invocations'


# CURL コマンドの文字列を生成
# (pythonであればrequestsライブラリを使用するのが一般的。ここではデモのためcurlコマンドを使用する)
cmd = f'''curl -X POST -u token:{token} {url} -H 'Content-Type: application/json' -d '{data_json}' '''
print(f'cmd => {cmd}')


# COMMAND ----------

# スコアリング結果がレスポンスとして返ります。

# 例:  [{"0": 137.89443969726562}, {"0": 160.4019012451172}, {"0": 136.89523315429688}]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC **注意**
# MAGIC 
# MAGIC モデルサービング機能を使用するとクラスタが起動し続けます。
# MAGIC 使用後は、必ずモデルサービングのUIからサービングをSTOPしてください。

# COMMAND ----------

# MAGIC %md ## Clean up
# MAGIC TensorBoardを終了するには:
# MAGIC - Databricks Runtime 7.1 MLもしくはそれ以前のversion => 下のセルのコメントを外して実行する
# MAGIC - Databricks Runtime 7.1 MLもしくはそれ以降のversion => クラスタからnotebookをデタッチする

# COMMAND ----------

#dbutils.tensorboard.stop()
