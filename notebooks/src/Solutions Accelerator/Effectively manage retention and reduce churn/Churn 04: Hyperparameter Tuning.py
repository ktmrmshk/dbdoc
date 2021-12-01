# Databricks notebook source
# MAGIC %md このノートブックの目的は、候補モデルに関連するハイパーパラメータを調整して、最適な構成を得ることです。 このノートブックは、Databricks ML 7.1+と**GPUベース**のノードを利用したクラスタ上で実行する必要があります。

# COMMAND ----------

# MAGIC %md ###ステップ1：データの読み込みと変換
# MAGIC 
# MAGIC はじめに、データの再読み込みを行い、欠損データ、カテゴリー値、フィーチャーの標準化に関する問題を解決するために、フィーチャーに変換を適用します。 このステップは、前回のノートブックで紹介・説明した作業の繰り返しです。

# COMMAND ----------

# DBTITLE 1,必要なライブラリの取り込み
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

import numpy as np

import time

# COMMAND ----------

# DBTITLE 1,Load Features & Labels
# retreive training & testing data
train = spark.sql('''
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
    b.ratio_number_total_after_exp_to_days_after_exp_with_session,
    c.is_churn
  FROM kkbox.train_trans_features a
  INNER JOIN kkbox.train_act_features b
    ON a.msno=b.msno
  INNER JOIN kkbox.train c
    ON a.msno=c.msno
  ''').toPandas()

test = spark.sql('''
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
    b.ratio_number_total_after_exp_to_days_after_exp_with_session,
    c.is_churn
  FROM kkbox.test_trans_features a
  INNER JOIN kkbox.test_act_features b
    ON a.msno=b.msno
  INNER JOIN kkbox.test c
    ON a.msno=c.msno
  ''').toPandas()

# split into features and labels
X_train_raw = train.drop(['msno','is_churn'], axis=1)
y_train = train['is_churn']

X_test_raw = test.drop(['msno','is_churn'], axis=1)
y_test = test['is_churn']

# COMMAND ----------

# DBTITLE 1,特徴量の変換
# replace missing values
impute = ColumnTransformer(
  transformers=[('missing values', SimpleImputer(strategy='most_frequent'), ['last_payment_method', 'city', 'gender', 'registered_via', 'bd'])],
  remainder='passthrough'
  )

# encode categoricals and scale all others
encode_scale =  ColumnTransformer( 
  transformers= [('ohe categoricals', OneHotEncoder(categories='auto', drop='first'), slice(0,4))], # features 0 through 3 should be the first four features imputed in previous step
  remainder= StandardScaler()  # standardize all other features
  )

# package transformation logic
transform = Pipeline([
   ('impute', impute),
   ('encode_scale', encode_scale)
   ])

# transform data
X_train = transform.fit_transform(X_train_raw)
X_test = transform.transform(X_test_raw)

# COMMAND ----------

# MAGIC %md ###ステップ2: ハイパーパラメータの調整 (XGBClassifier)
# MAGIC 
# MAGIC XGBClassifier は、モデルの学習を調整するための [多様なハイパーパラメータ](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier) を用意しています。 データとアルゴリズムの知識があれば、いくつかのハイパーパラメータを手動で設定することができます。しかし、ハイパーパラメータ間の相互作用は複雑なため、どの値の組み合わせが最良のモデル結果をもたらすかを正確に知ることは難しいでしょう。 このような場合には、異なるハイパーパラメータ設定で一連のモデル実行を行い、モデルの反応を観察して、最適な値の組み合わせを導き出すことになります。
# MAGIC 
# MAGIC hyperoptを使用すると、このタスクを自動化することができ、hyperoptフレームワークに探索するための潜在的な値の範囲を与えることができます。 モデルをトレーニングして評価指標を返す関数を呼び出すと、hyperoptは利用可能な検索空間を使って、値の最適な組み合わせを探します。
# MAGIC 
# MAGIC モデルの評価には、平均精度（AP）スコアを使用します。APスコアは、モデルが向上すると1.0に向かって増加します。 hyperoptは、評価指標が低下すると改善を認識するので、フレームワーク内の損失指標として-1 * APスコアを使用します。
# MAGIC 
# MAGIC これらをすべてまとめると、モデルの学習と評価の関数は次のようになります。

# COMMAND ----------

# DBTITLE 1,Hyperoptのモデル評価関数の定義
def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  X_train_input = X_train_broadcast.value
  y_train_input = y_train_broadcast.value
  X_test_input = X_test_broadcast.value
  y_test_input = y_test_broadcast.value  
  
  # configure model parameters
  params = hyperopt_params
  
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  params['tree_method']='gpu_hist'      # settings for running on GPU
  params['predictor']='gpu_predictor'   # settings for running on GPU
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_input, y_train_input)
  
  # predict
  y_prob = model.predict_proba(X_test_input)
  
  # score
  model_ap = average_precision_score(y_test_input, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)  # record actual metric with mlflow run
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md モデル評価関数の最初の部分では、トレーニング用とテスト用の特徴量とラベル・セットの複製をメモリから取り出します。 私たちの目的は、SparkTrialsとhyperoptを組み合わせて、Sparkクラスタ全体でモデルのトレーニングを並列化することです。これにより、複数のモデルのトレーニング評価を同時に実行し、検索空間を移動するのに必要な全体の時間を短縮することができます。 データセットをクラスターのワーカーノードに複製することで、トレーニングと評価に必要なデータのコピーを、最小限のネットワーク・オーバーヘッドで効率的に機能に利用できるようになります（次のセルで実行するタスク）。
# MAGIC 
# MAGIC **注**データ配布のオプションについては、Distributed Hyperopt [best practices documentation](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-best-practices.html#handle-datasets-of-different-orders-of-magnitude-notebook)を参照してください。

# COMMAND ----------

# DBTITLE 1,トレーニングとテストのデータセットをクラスタワーカーに複製する
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)

# COMMAND ----------

# MAGIC %md hyperoptによって関数に渡されるハイパーパラメータの値は、次のセルで定義される探索空間から得られます。 探索空間内の各ハイパーパラメータは、辞書の項目を使って定義され、その名前はハイパーパラメータを特定し、値はそのパラメータの潜在的な値の範囲を定義します。 hp.choice*を使って定義された場合、パラメータは定義済みの値のリストから選択されます。 hp.loguniform*で定義された場合、値は、連続した値の範囲から生成されます。 hp.quniform*を使って定義された場合、値は連続した範囲から生成されますが、範囲定義の第3引数で指定された精度のレベルに切り捨てられます。 hyperoptのハイパーパラメータ探索空間は、ライブラリの[オンラインドキュメント](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions)に示されているように、他の多くの方法で定義することができます。 

# COMMAND ----------

# DBTITLE 1,探索空間の定義
# define minimum positive class scale factor (as shown in previous notebook)
weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
  )
scale = weights[1]/weights[0]

# define hyperopt search space
search_space = {
    'max_depth' : hp.quniform('max_depth', 1, 30, 1)                                  # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
    ,'learning_rate' : hp.loguniform('learning_rate', np.log(0.01), np.log(0.40))     # learning rate for XGBoost
    ,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
    ,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
    ,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
    ,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
    ,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
    ,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
    ,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(scale), np.log(scale * 10))   # weight to assign positive label to manage imbalance
    }

# COMMAND ----------

# MAGIC %md モデル評価関数の残りの部分は非常に簡単です。 モデルの訓練と評価を行い、損失値（例：* -1 * AP Score）を hyperopt が解釈可能な辞書の一部として返します。 返された値に基づいて、hyperoptは、探索空間の定義内からハイパーパラメータ値の新しいセットを生成し、それを使ってメトリックの改善を試みます。hyperoptの評価数は、実行したいくつかのトレイルラン（図示せず）に基づいて、250に制限します。 潜在的な探索空間が大きいほど、また、モデル（トレーニングデータセットとの組み合わせ）が異なるハイパーパラメータの組み合わせに反応する度合いによって、hyperoptが局所的に最適な値に到達するために必要な反復回数が決まります。 hyperoptの出力を見ると、各評価の過程で損失指標がゆっくりと改善されていく様子がわかります。
# MAGIC 
# MAGIC **注意** XGBClassifierは、*evaluate_model*関数内で**GPU**を使用するように設定されています。これを**GPUベースのクラスタ**で実行していることを確認してください。

# COMMAND ----------

# perform evaluation
with mlflow.start_run(run_name='XGBClassifer'):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=250,
    trials=SparkTrials(parallelism=4), # THIS VALUE IS ALIGNED WITH THE NUMBER OF WORKERS IN MY GPU-ENABLED CLUSTER (guidance differs for CPU-based clusters)
    verbose=True
    )

# COMMAND ----------

# MAGIC %md チューニングの練習が終わったので、featuresとlabelsのデータセットの複製をリリースしましょう。 これでクラスタのリソースを圧迫することなく、作業を進めることができます。

# COMMAND ----------

# DBTITLE 1,複製されたデータセットの公開
# release the broadcast datasets
X_train_broadcast.unpersist()
X_test_broadcast.unpersist()
y_train_broadcast.unpersist()
y_test_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md ここで、hyperoptによって得られたハイパーパラメータの値を調べてみましょう。

# COMMAND ----------

# DBTITLE 1,最適化されたハイパーパラメータの設定
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md [hyperoptによって自動的に[mlflowに取り込まれた](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration.html)異なるモデル実行の結果を比較すると、いくつかのパラメータ設定がモデルのパフォーマンスに明らかな影響を与えることがわかります。比較チャートを見ることで、hyperoptが上記の設定に至った経緯を理解することができます。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/churn_xgbclassifier_hyperparams.PNG">
# MAGIC 
# MAGIC 最適化されたハイパーパラメータを使用して、mlflowで[persistence](https://docs.databricks.com/applications/mlflow/models.html)のモデルをトレーニングすることができます。 CPUベースの計算**に合わせて、ツリー法と予測パラメータを切り替えていることに注目してください。後日、CPUベースのインフラに展開するためにモデルをパッケージ化しますが、この変更はそのステップにモデルを合わせるものです。これはモデルの出力には影響しませんが、処理速度には影響します。
# MAGIC 
# MAGIC **注意** ノートブックの中で後で簡単に検索できるように、このモデルと他の最終モデルのランのmlflowランIDを保持するリストを定義しています。 このリストは、このノートブックの最後に、投票アンサンブルモデルに含めるために、永続化された各モデルを取得するために使用されます。

# COMMAND ----------

# DBTITLE 1,最終的なモデル情報を保持する変数の定義
# define list to hold run ids for later retrieval
run_ids = []

# COMMAND ----------

# DBTITLE 1,XGBClassifierモデルの学習
# train model with optimal settings 
with mlflow.start_run(run_name='XGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
   
  # configure params
  params = space_eval(search_space, argmin)
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])       
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight'])
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step'])
  if 'scale_pos_weight' in params: params['scale_pos_weight']=int(params['scale_pos_weight'])    
  params['tree_method']='hist'        # modified for CPU deployment
  params['predictor']='cpu_predictor' # modified for CPU deployment
  mlflow.log_params(params)
  
  # train
  model = XGBClassifier(**params)
  model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model')  # persist model with mlflow
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###Step 3: HistGradientBoostingClassifier & MLPClassiferモデルの学習
# MAGIC 
# MAGIC 前述のステップと同じ手法を用いて、HistGradientBoostingClassifierモデルの学習に最適なパラメータセットを特定しました（ここでは簡潔にするために省略します）。 これで、このモデルの最終的な学習を行うことができます。

# COMMAND ----------

# DBTITLE 1,HistGradientBoostingClassifierの学習
# set optimal hyperparam values
params = {
 'learning_rate': 0.046117525858818814,
 'max_bins': 129.0,
 'max_depth': 7.0,
 'max_leaf_nodes': 44.0,
 'min_samples_leaf': 39.0,
 'scale_pos_factor': 1.4641157666623326
 }

# compute sample weights
sample_weights = compute_sample_weight(
  'balanced', 
  y=y_train
  )

# train model based on these hyper params
with mlflow.start_run(run_name='HGB Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  # configure
  mlflow.log_params(params)
  params['max_depth'] = int(params['max_depth'])
  params['max_bins'] = int(params['max_bins'])
  params['min_samples_leaf'] = int(params['min_samples_leaf'])
  params['max_leaf_nodes'] = int(params['max_leaf_nodes'])
  sample_weights_factor = params.pop('scale_pos_factor')
  
  # train
  model = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000, **params)
  model.fit(X_train, y_train, sample_weight = sample_weights * sample_weights_factor)
  mlflow.sklearn.log_model(model, 'model')
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP Score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ニューラルネットワークにも同様の作業を行ったので、今度はニューラルネットワークのトレーニングを行います。

# COMMAND ----------

# DBTITLE 1,Train MLP Classifier
# optimal param settings
params = {
  'activation': 'logistic',
  'hidden_layer_1': 100.0,
  'hidden_layer_2': 35.0,
  'hidden_layer_cutoff': 15,
  'learning_rate': 'adaptive',
  'learning_rate_init': 0.3424456484117518,
  'solver': 'sgd'
   }

# train model based on these params
with mlflow.start_run(run_name='MLP Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  mlflow.log_params(params)
  
  # hidden layer definitions
  hidden_layer_1 = int(params.pop('hidden_layer_1'))
  hidden_layer_2 = int(params.pop('hidden_layer_2'))
  hidden_layer_cutoff = int(params.pop('hidden_layer_cutoff'))
  if hidden_layer_2 > hidden_layer_cutoff:
    hidden_layer_sizes = (hidden_layer_1, hidden_layer_2)
  else:
    hidden_layer_sizes = (hidden_layer_1)
  params['hidden_layer_sizes']=hidden_layer_sizes
  
  # train
  model = MLPClassifier(max_iter=10000, **params)
  model.fit(X_train, y_train)
  mlflow.sklearn.log_model(model, 'model')
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP Score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###ステップ4： 投票型分類器の学習
# MAGIC 
# MAGIC これで、最適化された3つのモデルができあがりました。投票アンサンブルを使用して、これら3つのモデルの予測を組み合わせて、どのモデルよりも優れた新しい予測を作成することができます。まず最初に、前のステップで学習したモデルを取得する必要があります。

# COMMAND ----------

# DBTITLE 1,学習したモデルの取得
models = []

# for each final training run, retreive its model from mlflow 
for run_id in run_ids:
  models += [(run_id[0], mlflow.sklearn.load_model('runs:/{0}/model'.format(run_id[1])))] 

models

# COMMAND ----------

# MAGIC %md 次に、これらのモデルの入力をどのように組み合わせるかを検討する必要があります。 デフォルトでは、各モデルは同じように考慮されますが、それぞれに割り当てられた重みを調整することで、少しでも良いスコアを得ることができるかもしれません。 重みを探索空間の一部として定義すれば、どの重みが最良の結果をもたらすかの検討をhyperoptで自動化することができます。 組み合わせた重みが1になるように重みを定義していないことに注意してください。 その代わりに、投票アンサンブルに比例した重み付けの計算を実行させます。

# COMMAND ----------

# DBTITLE 1,探索空間の構築
search_space = {}

# for each model, define a weight hyperparameter
for i, model in enumerate(models):
    search_space['weight{0}'.format(i)] = hp.loguniform('weight{0}'.format(i), np.log(0.0001), np.log(1.000))

search_space

# COMMAND ----------

# MAGIC %md 前述のように、評価関数を定義します。 データセットの複製版へのアクセスに加えて、後のステップで定義するmodels_broadcast変数を通じて、学習済みモデルの複製版にもアクセスしていることに注意してください。 ワーカーノードへのブロードキャストは、データセットに限定する必要はありません。
# MAGIC 
# MAGIC 各モデルは事前に学習されているので、投票分類器の *fit()* メソッドを呼び出す必要はありません。HistGradientBoostingClassifierのフィッティングには、sample_weightsパラメータを渡す必要があるため、この呼び出しを省略することは重要です。このパラメータは、必要に応じて *fit()* メソッドに渡すと、他の2つのモデルに拒否されてしまいます。 メソッドの呼び出しをバイパスすることはsklearnでは明示的にサポートされていませんが、[hack](https://stackoverflow.com/questions/42920148/using-sklearn-voting-ensemble-with-partial-fit)を使えば、このステップをスキップしても予測を行うようにモデルを騙すことができます。

# COMMAND ----------

# DBTITLE 1,アンサンブル評価関数の定義
def evaluate_model(hyperopt_params):
  
  # accesss replicated input data
  X_train_input = X_train_broadcast.value
  y_train_input = y_train_broadcast.value
  X_test_input = X_test_broadcast.value
  y_test_input = y_test_broadcast.value  
  models_input = models_broadcast.value  # pre-trained models
    
  # compile weights parameter used by the voting classifier (configured for 10 models max)
  weights = []
  for i in range(0,10):
    if 'weight{0}'.format(i) in hyperopt_params:
      weights += [hyperopt_params['weight{0}'.format(i)]]
    else:
      break
  
  # configure basic model
  model = VotingClassifier(
      estimators = models_input, 
      voting='soft',
      weights=weights
      )

  # configure model to recognize child models as pre-trained 
  clf_list = []
  for clf in models_input:
    clf_list += [clf[1]]
  model.estimators_ = clf_list
  
  # predict
  y_prob = model.predict_proba(X_test_input)
  
  # score
  model_ap = average_precision_score(y_test_input, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  # invert metric for hyperopt
  loss = -1 * model_ap  
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md 探索空間とモデル評価関数が定義されたので、投票アンサンブルの最適な重みのセットを見つけるために反復することができます。

# COMMAND ----------

# DBTITLE 1,アンサンブルハイパーパラメータの最適化
X_train_broadcast = sc.broadcast(X_train)
X_test_broadcast = sc.broadcast(X_test)
y_train_broadcast = sc.broadcast(y_train)
y_test_broadcast = sc.broadcast(y_test)
models_broadcast = sc.broadcast(models)

# perform evalaution
with mlflow.start_run(run_name='Voting: {0}'.format('weights')):
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,
    max_evals=250,
    trials=SparkTrials(parallelism=4),
    verbose=True
    )
  
# release the broadcast dataset
X_train_broadcast.unpersist()
X_test_broadcast.unpersist()
y_train_broadcast.unpersist()
y_test_broadcast.unpersist()
models_broadcast.unpersist()

# COMMAND ----------

# DBTITLE 1,最適化されたハイパーパラメータの設定を表示
space_eval(search_space, argmin)

# COMMAND ----------

# MAGIC %md 3つのモデルを組み合わせることで、どのモデルでも単独でのAPスコアよりも少し良いスコアを得ることができました。 繰り返しになりますが、hyperoptによって得られた重みは、個々のモデルに適用された実際の重みではなく、比例した重みを計算するための出発点であることに注意することが重要です。 このため、mlflowの比較チャートを使って、hyperoptがどのように重みを計算するかを調べるのは、かなり難しいことです。
# MAGIC 
# MAGIC しかし、それはさておき、最終的な投票アンサンブルモデルを訓練し、後で再利用できるように保存することができます。

# COMMAND ----------

# DBTITLE 1,アンサンブルモデルの学習
params = space_eval(search_space, argmin)

with mlflow.start_run(run_name='Voting Final Model') as run:
  
  # capture run info for later use
  run_id = run.info.run_id
  run_name = run.data.tags['mlflow.runName']
  run_ids += [(run_name, run_id)]
  
  mlflow.log_params(params)
  
  # compile weights (configured for 10 max)
  weights = []
  for i in range(0,10):
    if 'weight{0}'.format(i) in params:
      weights += [params['weight{0}'.format(i)]]
    else:
      break
  
  # configure basic model
  model = VotingClassifier(
      estimators = models, 
      voting='soft',
      weights=weights
      )

  # configure model to recognize child models are pre-trained 
  clf_list = []
  for clf in models:
    clf_list += [clf[1]]
  model.estimators_ = clf_list
  
  # predict
  y_prob = model.predict_proba(X_test)
  
  # score
  model_ap = average_precision_score(y_test, y_prob[:,1])
  mlflow.log_metric('avg precision', model_ap)
  
  print('Model logged under run_id "{0}" with AP score of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ###Step 5: モデルパイプラインの永続化
# MAGIC 
# MAGIC これで最適化されたモデルがトレーニングされ、パーシストされました。 しかし、このモデルは変換前のデータを想定して作られています。 このモデルを使ってビジネスの予測を行うことを考えると、このノートの冒頭で説明したデータ変換のステップをモデルと組み合わせて、変換されていない特徴データを直接渡せるようにすることが有効です。 この問題に取り組むために、先に定義したColumnTransformersと、最後のステップでトレーニングしたVoting Classifierモデルを、統一されたモデルパイプラインに組み合わせます。

# COMMAND ----------

# DBTITLE 1,モデルパイプラインの組み立て
# assemble pipeline
model_pipeline = Pipeline([
   ('impute', impute),
   ('encode_scale', encode_scale),
   ('voting_ensemble', model)
   ])

# COMMAND ----------

# MAGIC %md パイプラインを定義する際には、通常、各ステップを学習するために*fit()*メソッドを呼び出しますが、このノートブックでは、各ステップが既に様々な時点で学習されているので、直接予測に移ることができます。 パイプラインが期待通りに動作することを確認するために、**生の**テストデータを渡し、評価指標を計算して、最後のステップで観測されたものと同じであることを確認します。

# COMMAND ----------

# DBTITLE 1,モデルパイプラインの動作確認 
# predict
y_prob = model_pipeline.predict_proba(X_test_raw)
  
# score
model_ap = average_precision_score(y_test, y_prob[:,1])

print('AP score: {0:.5f}'.format(model_ap))

# COMMAND ----------

# MAGIC %md すべて順調です。 このモデルを後で再利用するために保存する準備ができたところですが、最後に克服しなければならない課題があります。
# MAGIC 
# MAGIC このモデルをmlflowで保存し、場合によってはSparkにpandas UDFとして登録することになります。 mlflowでのこのような関数のデフォルトの配置は、pandas UDFをモデルの*predict()*メソッドにマッピングします。 前回のノートブックで覚えているかもしれませんが、*predict()*メソッドは、50%の確率のしきい値に基づいて0または1のクラス予測を返します。 mlflowのサービングメカニズムに登録した際に、モデルが実際の正のクラス確率を返すようにしたい場合は、*predict()*メソッドをオーバーライドするカスタマーラッパーを書く必要があります。

# COMMAND ----------

# DBTITLE 1,Predictメソッドをオーバーライドするラッパーの定義
# shamelessly stolen from https://docs.databricks.com/_static/notebooks/mlflow/mlflow-end-to-end-example-aws.html

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# COMMAND ----------

# MAGIC %md これで、モデルを永続化することができます。その際、カスタムラッパーがコールに含まれていることを確認してください。

# COMMAND ----------

# DBTITLE 1,Persist Model Pipeline 
with mlflow.start_run(run_name='Final Pipeline Model') as run:
  
  run_id = run.info.run_id
  
  # record the score with this model
  mlflow.log_metric('avg precision', model_ap)
  
  # persist the model with the custom wrapper
  wrappedModel = SklearnModelWrapper(model_pipeline)
  mlflow.pyfunc.log_model(
    artifact_path='model', 
    python_model=wrappedModel
    )
  
print('Model logged under run_id "{0}" with log loss of {1:.5f}'.format(run_id, model_ap))

# COMMAND ----------

# MAGIC %md ここで、[mlflow model registry](https://www.mlflow.org/docs/latest/model-registry.html)機能を使って、このモデルを本番用*のインスタンスにしてみましょう。 一般的なMLOpsのワークフローでは、モデルの初期登録からステージング、そして本番への移行には、複数のチームメンバーと、本番への対応を確認するための一連のテストが必要になります。 今回のデモでは、モデルを本番環境に直接移行し、他の本番環境のインスタンスをアーカイブすることで、次のノートブックでのモデルの検索が簡単になるようにします。

# COMMAND ----------

model_name = 'churn-ensemble'

# archive any production model versions (from any previous runs of this notebook or manual workflow management)
client = mlflow.tracking.MlflowClient()
for mv in client.search_model_versions("name='{0}'".format(model_name)):
    # if model with this name is marked production
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=mv.name,
        version=mv.version,
        stage='archived'
        )
      
# register last deployed model with mlflow model registry
mv = mlflow.register_model(
    'runs:/{0}/model'.format(run_id),
    'churn-ensemble'
    )
model_version = mv.version

# wait until newly registered model moves from PENDING_REGISTRATION to READY status
while mv.status == 'PENDING_REGISTRATION':
  time.sleep(5)
  for mv in client.search_model_versions("run_id='{0}'".format(run_id)):  # new search functionality in mlflow 1.10 will make easier
    if mv.version == model_version:
      break
      
# transition newly deployed model to production stage
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )      
