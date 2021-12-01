# Databricks notebook source
# MAGIC %md このノートブックの目的は、解約予測のための代替モデルを検討することです。  このノートブックは、**Databricks ML 7.1+**と**CPUベース**のノードを活用したクラスタ上で実行する必要があります。

# COMMAND ----------

# MAGIC %md ###Step 1: 特徴量とラベルの準備
# MAGIC 
# MAGIC 最初のステップは、モデルの学習と評価に使用する特徴量とラベルを取得することです。 データ準備のロジックについては、前々回のノートで説明しました。

# COMMAND ----------

# DBTITLE 1,必要なライブラリの取り込み
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, log_loss, precision_recall_curve, auc, average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# COMMAND ----------

# DBTITLE 1,特徴量とラベルを取得する
# retrieve training dataset
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

# retrieve training dataset
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


# separate features and labels
X_train_raw = train.drop(['msno','is_churn'], axis=1)
y_train = train['is_churn']

# separate features and labels
X_test_raw = test.drop(['msno','is_churn'], axis=1)
y_test = test['is_churn']

# COMMAND ----------

# MAGIC %md *X_train_raw*と*X_test_raw*という特徴量セットは、欠損値やカテゴリー値に対応するために変換する必要があります。 また、評価するモデルの要件に合わせて、連続的な特徴量を調整する必要があります。

# COMMAND ----------

# DBTITLE 1,特徴量データの変換
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

# apply transformations
X_train = transform.fit_transform(X_train_raw)
X_test = transform.transform(X_test_raw)

# COMMAND ----------

# MAGIC %md ###Step 2: 評価指標の検討
# MAGIC 
# MAGIC 解約予測は、一般的に、解約した顧客やサブスクリプションを*ポジティブクラス*（解約ラベル1）、解約していない顧客やサブスクリプションを*ネガティブクラス*（解約ラベル0）とする2値分類問題として扱われます。 別の言い方をすれば、ポジティブなクラスは「少数派」であり、ネガティブなクラスは「多数派」であるということです。 ビジネスにとっては素晴らしいことですが、少数派クラスと多数派クラスの間の不均衡は、予測解を学習する任務を負ったアルゴリズムに問題を生じさせます。
# MAGIC 
# MAGIC これを理解するために、トレーニングデータセットとテストデータセットで、否定的な解約イベントと肯定的な解約イベントのバランスが悪いことを調べてみましょう。

# COMMAND ----------

# DBTITLE 1,クラスラベルのバランスを検証する
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   CONCAT(is_churn,' - ',CASE WHEN is_churn=0 THEN 'not churned' ELSE 'churned' END) as class,
# MAGIC   dataset,
# MAGIC   count(*) as instances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     is_churn, 'train' as dataset
# MAGIC   FROM kkbox.train
# MAGIC   UNION ALL
# MAGIC   SELECT
# MAGIC     is_churn, 'test' as dataset
# MAGIC   FROM kkbox.test
# MAGIC   ) 
# MAGIC GROUP BY is_churn, dataset
# MAGIC ORDER BY is_churn DESC

# COMMAND ----------

# MAGIC %md この2つの期間中、96～97%の加入者が解約していないため、単純に**すべてのインスタンス**に**解約していない**というラベルを付けるだけのナイーブモデルでは、非常に高い精度のスコアを達成することができます。

# COMMAND ----------

# DBTITLE 1,ナイーブモデルの精度を評価する
# generate naive churn prediction of ALL negative class
naive_y_pred = np.zeros(y_test.shape[0], dtype='int32')

print('Naive model Accuracy:\t{0:.6f}'.format( accuracy_score(y_test, naive_y_pred)))

# COMMAND ----------

# MAGIC %md 一見、かなり良いスコアが出ているように見えますが、このモデルでは、解約している顧客を特定するという目的に沿っていないことがわかります。 では、学習したモデルとの比較を見てみましょう。

# COMMAND ----------

# DBTITLE 1,予測モデルの学習
# train the model
trained_model = LogisticRegression(max_iter=1000)
trained_model.fit(X_train, y_train)

# predict
trained_y_pred = trained_model.predict(X_test)

# calculate accuracy
print('Trained Model Accuracy:\t{0:.6f}'.format(accuracy_score(y_test, trained_y_pred)))

# COMMAND ----------

# MAGIC %md アンバランスなシナリオでは、精度ではモデルの性能を十分に評価できません。 ほとんどスキルのないモデルでも、多数派クラスの予測に頼るだけで高い精度を得ることができます。モデル評価のより良いアプローチは、正のクラスを予測するモデルの能力を検証するメトリクスを活用することです。なぜなら、正のクラスは解約予測のシナリオにおいてより高い価値や重要性を持つと考えられるからです。 この種の評価には、[Precision, recall, F1 score](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)がよく使われます。
# MAGIC 
# MAGIC しかし、これらの指標はすべてを語るものではありません。それぞれの指標は、予測されたクラスの割り当て、つまり0か1に基づいて計算されています。 蓋を開けてみれば、モデルは実際にはクラスの確率を予測しており、その確率は単純なクラスの割り当てよりもはるかに多くの情報を提供しています。

# COMMAND ----------

# DBTITLE 1,学習したモデルの確率を取得する
# predict class assignment probabilities
trained_y_prob = trained_model.predict_proba(X_test)
trained_y_prob[110:120]

# COMMAND ----------

# MAGIC %md 類似確率は、訓練データセットにおける各クラスの割合を用いて、ナイーブモデルに割り当てられるかもしれません。

# COMMAND ----------

# DBTITLE 1,ナイーブモデルの確率を取得する
# calculate ratio of negative class instances in training dataset
label_counts = np.unique(y_train, return_counts=True)[1]
negclass_prop = label_counts[0]/np.sum(label_counts)

# construct a set of class probabilies representing class proportions in training set
naive_y_prob = np.empty(trained_y_prob.shape, dtype='float')
naive_y_prob[:] = [negclass_prop, 1-negclass_prop]

# display results
naive_y_prob

# COMMAND ----------

# MAGIC %md クラスの割り当てを決定するために、クラスの確率にしきい値を適用します。一方のクラスがしきい値を上回っていれば、どちらのクラスのラベルがインスタンスに割り当てられるかが決まります。前述のF1、precision、recallといった指標では、しきい値を設定し、ラベルの割り当てに基づいて指標を計算する必要があります。 predict()*メソッドでは50%の閾値を使用していますが、任意の閾値を用いてラベルを割り当てることも可能です。
# MAGIC 
# MAGIC より根本的な問題は、適切なしきい値が何であるかがまだわからないことです。 また、閾値は、顧客の収益/利益の可能性やその他の要因に基づいて、顧客のインスタンスによって異なる可能性があります。 我々の目的が、潜在的なしきい値の範囲における予測能力に基づいてモデルを評価することである場合、実際のクラスラベルに対する予測確率を調べるメトリクスを検討することができます。 最も一般的な指標の1つがROC AUCスコアです。
# MAGIC 
# MAGIC **注** この指標や類似の指標では、正のクラスの確率のみが必要です。正のクラスに割り当てられた確率と負のクラスに割り当てられた確率を足すと、常に1.0になります。

# COMMAND ----------

# DBTITLE 1,ROC AUCを評価する
# calculate ROC AUC for trained & naive models
trained_auc = roc_auc_score(y_test, trained_y_prob[:,1])
naive_auc = roc_auc_score(y_test, naive_y_prob[:,1])

print('Trained ROC AUC:\t{0:.6f}'.format(trained_auc))
print('Naive ROC AUC:\t\t{0:.6f}'.format(naive_auc))

# COMMAND ----------

# MAGIC %md ROC AUCスコアは、受信者操作曲線の下の面積を測定します。受信者演算子曲線（ROC）は、あるインスタンスを陰性クラスまたは陽性クラスのいずれかのメンバーとして識別する確率のしきい値を増加させたときの、真の陽性予測と偽陽性予測のバランスの変化をプロットしたものです。この曲線の下の領域を測定することで、閾値の全範囲におけるモデルの予測能力を特定することができます。 ナイーブモデルに割り当てられた0.50のスコアは、予測能力がないことを示しています。一方、ロジスティック回帰モデルに割り当てられた約90%のスコアは、50%の閾値で2つのモデルが非常に似た予測を行う（ほぼ同じ精度のスコアに反映されている）としても、その予測の確実性がはるかに高いことを示しています。 しかし、完璧な予測はROC AUCスコアが最大で1.00となるため、改善の余地があります。
# MAGIC 
# MAGIC 曲線（およびその下の面積）を視覚化することで、曲線下面積（AUC）の概念を少し理解しやすくなります。
# MAGIC 
# MAGIC **注** 以下のグラフでは、ナイーブモデルのAUCは紫色の陰影が付けられていますが、学習済みモデルのAUCは赤と紫の陰影が付いた部分の両方を含んでいます。

# COMMAND ----------

# DBTITLE 1,Visualize ROC AUC
trained_fpr, trained_tpr, trained_thresholds = roc_curve(y_test, trained_y_prob[:,1])
naive_fpr, naive_tpr, naive_thresholds = roc_curve(y_test, naive_y_prob[:,1])

# define the plot
fig, ax = plt.subplots(figsize=(10,8))

# plot the roc curve for the model
plt.plot(naive_fpr, naive_tpr, linestyle='--', label='Naive', color='xkcd:eggplant')
plt.plot(trained_fpr, trained_tpr, linestyle='solid', label='Trained', color='xkcd:cranberry')

# shade the area under the curve
ax.fill_between(trained_fpr, trained_tpr, 0, color='xkcd:light pink')
ax.fill_between(naive_fpr, naive_tpr, 0, color='xkcd:dusty lavender')

# label each curve with is ROC AUC score
ax.text(.55, .3, 'Naive AUC:  {0:.6f}'.format(naive_auc), fontsize=14)
ax.text(.32, .5, 'Trained AUC:  {0:.6f}'.format(trained_auc), fontsize=14)

# adjust the axes to intersect at (0,0) and (1,1)
ax.spines['left'].set_position(('data', 0.0))
ax.axes.spines['bottom'].set_position(('data', 0.0))
ax.axes.spines['right'].set_position(('data', 1.0))
ax.axes.spines['top'].set_position(('data', 1.0))

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# show the legend
plt.legend(loc=(0.05, 0.85))

# show the plot
plt.show()

# COMMAND ----------

# MAGIC %md 予測された確率からROC AUCを計算することで、ある閾値では似たような予測をしているように見える2つのモデルの違いを調べることができます。しかし、多くの研究者は、不均衡なシナリオ（解約予測など）では、ROC AUCがモデル評価の信頼できる根拠にならないことを指摘しています。その理由を理解するには、[このホワイトペーパー](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349800/)を読む価値がありますが、簡単に言うと、ROCのx軸に沿ってプロットされた偽陽性率は、陽性クラスのインスタンスの発生率が低いため、比較的感度が低いということです。 言い換えれば、予測がネガティブクラスに大きく偏っている場合、モデルは多くの偽陽性予測を行うことはありません。
# MAGIC 
# MAGIC 別の方法として、潜在的なしきい値の範囲で精度とリコールをプロットする同様の曲線を調べることができます。精度は、正のクラスの予測がどのくらいの割合で正しいかを示し、正の予測の精度と表現することができます。 Recallは、データセット内の正のクラスのインスタンスのうち、モデルによって識別されたものの割合を示し、予測の完全性を測定します。 この2つの指標を潜在的なしきい値の範囲でプロットすると、ROC曲線に似た曲線が得られますが、すべての正のクラスを識別しようとすると、正のクラス予測の正解率がどのように低下するかがわかります。また、ROC曲線と同様に、PRC（precision-recall curve）を曲線下の面積計算（PRC AUC）で要約することができます。 
# MAGIC 
# MAGIC しかし、PRC曲線に対して計算した場合、AUCは楽観的すぎる可能性があるため、その代わりに[平均精度スコア](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) (APスコア)を使用することが提案されています。 APスコアは、しきい値間のリコールの変化を加重要素として、しきい値の範囲にわたる加重平均精度を計算します。 APスコアは、AUCに似たスコアを提供しますが、もう少し保守的になる傾向があります。

# COMMAND ----------

# DBTITLE 1,APスコアを評価する
trained_ap = average_precision_score(y_test, trained_y_prob[:,1])
naive_ap = average_precision_score(y_test, naive_y_prob[:,1])

print('Naive AP:\t{0:.6f}'.format(naive_ap))
print('Trained AP:\t{0:.6f}'.format(trained_ap))

# COMMAND ----------

# MAGIC %md 精度と再現性の関係をよりよく理解し、PRC APスコアをどのように解釈すべきかを理解するために、さまざまな潜在的しきい値に対する精度と再現性のプロットを見てみましょう。

# COMMAND ----------

# DBTITLE 1,Visualize Precision-Recall
# get values for PR curve
naive_precision, naive_recall, naive_thresholds = precision_recall_curve(y_test, naive_y_prob[:,1])
naive_thresholds = np.append(naive_thresholds, 1)
trained_precision, trained_recall, trained_thresholds = precision_recall_curve(y_test, trained_y_prob[:,1])
trained_thresholds = np.append(trained_thresholds, 1)

# define the plot
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# precision
ax[0].set_title('Precision')
ax[0].plot(trained_thresholds, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[0].fill_between(trained_thresholds, trained_precision, 0, color='xkcd:light grey green')

ax[0].spines['left'].set_position(('data', 0.0))
ax[0].spines['bottom'].set_position(('data', 0.0))
ax[0].spines['right'].set_position(('data', 1.0))
ax[0].spines['top'].set_position(('data', 1.0))

ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Precision')

# recall
ax[1].set_title('Recall')
ax[1].plot(trained_thresholds, trained_recall, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[1].fill_between(trained_thresholds, trained_recall, 0, color='xkcd:light grey green')

ax[1].spines['left'].set_position(('data', 0.0))
ax[1].spines['bottom'].set_position(('data', 0.0))
ax[1].spines['right'].set_position(('data', 1.0))
ax[1].spines['top'].set_position(('data', 1.0))

ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Recall')

# precision-recall curve

test_positive_prop = len(y_test[y_test==1]) / len(y_test)

ax[2].set_title('Precision-Recall')
ax[2].plot([0,1], [test_positive_prop,test_positive_prop], linestyle='--', label='Naive', color='xkcd:sunflower yellow')
ax[2].plot(trained_recall, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')

# shade the area under the curve
ax[2].fill_between(trained_recall, trained_precision, 0, color='xkcd:light grey green')
ax[2].fill_between([0,1], [test_positive_prop,test_positive_prop], 0, color='xkcd:buff')

# label each curve with is ROC AUC score
ax[2].text(.2, .075, 'Naive AP:  {0:.6f}'.format(naive_ap), fontsize=14)
ax[2].text(.3, .3, 'Trained AP:  {0:.6f}'.format(trained_ap), fontsize=14)

# adjust the axes to intersect at (0,0) and (1,1)
ax[2].spines['left'].set_position(('data', 0.0))
ax[2].axes.spines['bottom'].set_position(('data', 0.0))
ax[2].axes.spines['right'].set_position(('data', 1.0))
ax[2].axes.spines['top'].set_position(('data', 1.0))

# axis labels
ax[2].set_xlabel('Recall')
ax[2].set_ylabel('Precision')

# show the legend
ax[2].legend(loc=(0.75, 0.85))

# COMMAND ----------

# MAGIC %md 出力の左端にある精度曲線から始めると、一般的に、閾値を上げると、正のクラスを正確に予測するモデルの能力が高まることがわかります。 これは、証拠が強ければ強いほど、つまり閾値の要件が高ければ高いほど、正しい予測の割合が高くなるということです。 
# MAGIC 
# MAGIC では、0.8の閾値付近になると、精度はどうなるのでしょうか？ このしきい値を超えるとチャートが不安定になるのは、この範囲の確率でポジティブなクラスの予測が少ないことを反映していると思われます。 このノートの次のステップでは、なぜこのモデルが正のクラスのインスタンスをより確実に予測するのに苦労しているのかを検討します。
# MAGIC 
# MAGIC また、素朴なモデルの精度はどうでしょうか（プロットされていません）。私たちの素朴なモデルでは、すべてのメンバーが、訓練データセット全体で正クラスのメンバーが占める割合、約3%と同等の正クラス確率を持つとしています。 同じ3％の値のしきい値（～0.03）以下では、すべてのインスタンスが正クラスであると推測されますが、テストセットに含まれる正クラスのメンバーの割合とほぼ同じ約4％の確率でしか正解しません。このしきい値を超えると、正のクラスの予測ができなくなるので、計算すべき精度指標はありません。
# MAGIC 
# MAGIC ここで、モデルの予測によって識別できた実際の正のクラスメンバーの割合であるリコールを検討します。しきい値を0に設定すると、すべてのインスタンスが正のクラスであると想定されるため、すべてのインスタンスを捕捉することができます。 精度のグラフに戻ると、これらの予測のうち約4％しか正しくないことがわかります。つまり、しきい値が0のときには、すべての正のクラスのメンバーを識別することができますが、たくさんの偽の正の予測もしてしまうことになります。閾値を上げていくと、正のクラスのインスタンスをどんどん逃していき、正のクラスのインスタンスをほとんど捉えられなくなります。
# MAGIC 
# MAGIC 一方、ナイーブモデルはどうでしょうか？閾値0.03以下では、各インスタンスにほぼその値の確率を割り当てているため、ナイーブモデルはすべての正のインスタンスを捕捉します。しかし、0.03のしきい値を超えると、正のクラスの予測ができなくなり、すべての正のクラスのインスタンスを見逃してしまいます。
# MAGIC 
# MAGIC しきい値が上下したときに、精度と回収率の曲線がどのように振る舞うかを考えると、これらの2つのメトリクスの間には綱引きがあることがわかります。 しきい値を低くすると、より多くのポジティブなクラスインスタンスをキャプチャでき、リコールが増加しますが、多くの間違った予測を拾うことになり、精度が低下します。しきい値を高くすると、ポジティブなクラスの予測の精度は上がりますが、実際のポジティブなインスタンスをより多く予測することができません。
# MAGIC 
# MAGIC この関係はprecision-recall curve（PRC）で表されます。理想的な状況では、各クラスのインスタンスを100%の確実性で予測し、閾値に関係なく、精度が常に1.0、リコールが1.0となります。 現実の世界では、私たちの目標は、モデルをこの理想的な状態にどんどん近づけて、PRCをプロットエリアの右上隅に押し上げることです。 このグラフからわかるように、理想に近づけるためには、まだ多くの課題があります。この理想からどれくらい離れているかを要約するために、平均精度（AP）スコアを計算することができます。これはAUCと同様に、理想的なPRCに近づくにつれて1.00に近づきます。
# MAGIC 
# MAGIC では、素朴なモデルはPRCにどのように適合するのでしょうか？我々の素朴なモデルは、約3%(~0.03)以下のしきい値に対して約4%の精度を持っていることを覚えておいてください。 (繰り返しますが、この2つの値は、テストデータセットとトレーニングデータセットにおける正のクラスメンバーの割合をそれぞれ反映しています)。また、約3%以下のしきい値では100%のリコールが得られますが、しきい値がこのマークを超えると、正のインスタンスを識別できなくなるため、リコールは得られません。このことがPRC曲線に与える影響は、精度と想起率はナイーブなシナリオでは一定の関係にあり、テストクラスにおけるポジティブクラスのインスタンスの割合、約4%に設定された水平方向のバーで図示することができます。 この線は、識別されたしきい値以下では、精度とリコールが固定の関係にあるため、変化しません。

# COMMAND ----------

# MAGIC %md ##Step 3: クラスの重み付けとログロスの検討
# MAGIC 
# MAGIC APスコアはモデルがポジティブなクラス予測をするための総合的な能力を評価する手段となります。 ここでは、このスコアを主要な評価指標とします。 しかし、もう一つの重要な指標であるログロスを調べる必要があります。
# MAGIC 
# MAGIC 多くの機械学習アルゴリズムは、モデルのエラーを最小化するために反復的な最適化を行います。 ログロスは、機械学習アルゴリズムで使用される一般的なエラー計算の1つです。 ログロスとは、あるクラス（0または1）と、そのクラスに関連する予測確率とのギャップを計算するものです。 この指標の対数部分は、確率がインスタンスのクラス割り当てから離れていくにつれて、ペナルティが指数関数的に大きくなることを意味しています。
# MAGIC 
# MAGIC 対数損失を最小化することで、モデルは予測の確信度を高めていきます。 しかし、データセットにはネガティブクラスのインスタンスが不均衡に多く存在するため、ネガティブクラスの予測に自信を持てるようになれば、モデルは最大の報酬を得ることができます。 ポジティブクラスの予測が不確かな場合、ログロスのスコア全体にはほとんど影響しません。なぜなら、データセットに含まれるポジティブクラスの数が圧倒的に少ないからです。
# MAGIC 
# MAGIC 最低限、負のクラスと正のクラスのインスタンスを同等にして、ログロスを計算するモデルが必要です。 これは、負のクラスと正のクラスのインスタンスに関連するペナルティに、データセット内の他のクラスの割合を反映した重みを乗じることで実現できます。 これは、多くのMLモデルタイプの*class_weight*引数に「balanced」という文字列を与えることで実現できます。

# COMMAND ----------

# DBTITLE 1,ポジティブなクラスとネガティブなクラスを同じように考慮してモデルをトレーニングする
# train the model
balanced_model = LogisticRegression(max_iter=10000, class_weight='balanced')
balanced_model.fit(X_train, y_train)

# predict
balanced_y_prob = balanced_model.predict_proba(X_test)

# score
balanced_ap = average_precision_score(y_test, balanced_y_prob[:,1])

# calculate accuracy
print('Trained AP:\t\t{0:.6f}'.format(trained_ap))
print('Balanced Model AP:\t{0:.6f}'.format(balanced_ap))

# COMMAND ----------

# MAGIC %md 2つのクラスのバランスをとるために使用されたウェイトをより明確に確認するために、*compute_class_weight*ユーティリティを呼び出すことができます。このユーティリティは、前のモデル実行で「balanced」値を提供して生成されたものと同じ結果を返します。 最初の重みは、負のクラスに割り当てられ、その影響を減らします。 2つ目の重みは、ポジティブなクラスに割り当てられ、その影響力を増加させます。

# COMMAND ----------

weights = compute_class_weight(
  'balanced', 
  classes=np.unique(y_train), 
  y=y_train
  )

weights

# COMMAND ----------

# MAGIC %md バランスのとれたウェイト付けは、あくまでも出発点としての提案であることに留意する必要があります。 この重み付けは、多数派のクラスと少数派のクラスを同等にするものであり、前回の実行結果が示すように、これは常にモデルのスコアを向上させるものではなく、肯定的なクラス予測と否定的なクラス予測が同じように得意または不得意なモデルとなるものです。
# MAGIC 
# MAGIC クラスの重みのバランスをとることで何が起こるかを理解するために、上述の精度-再現曲線を見てみましょう。

# COMMAND ----------

# get values for PR curve
balanced_precision, balanced_recall, balanced_thresholds = precision_recall_curve(y_test, balanced_y_prob[:,1])
balanced_thresholds = np.append(balanced_thresholds, 1)


# define the plot
plt.clf()
fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=True, figsize=(32,10))

# precision
ax[0].set_title('Precision')
ax[0].plot(trained_thresholds, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[0].plot(balanced_thresholds, balanced_precision, linestyle='solid', label='Balanced', color='xkcd:blood orange')

ax[0].fill_between(trained_thresholds, trained_precision, 0, color='xkcd:light grey green')
ax[0].fill_between(balanced_thresholds, balanced_precision, 0, color='xkcd:pale salmon')

ax[0].spines['left'].set_position(('data', 0.0))
ax[0].spines['bottom'].set_position(('data', 0.0))
ax[0].spines['right'].set_position(('data', 1.0))
ax[0].spines['top'].set_position(('data', 1.0))

ax[0].set_xlabel('Threshold')
ax[0].set_ylabel('Precision')

# recall
ax[1].set_title('Recall')
ax[1].plot(trained_thresholds, trained_recall, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[1].plot(balanced_thresholds, balanced_recall, linestyle='solid', label='Balanced', color='xkcd:blood orange')

ax[1].fill_between(trained_thresholds, trained_recall, 0, color='xkcd:light grey green')
ax[1].fill_between(balanced_thresholds, balanced_recall, 0, color='xkcd:pale salmon')

ax[1].spines['left'].set_position(('data', 0.0))
ax[1].spines['bottom'].set_position(('data', 0.0))
ax[1].spines['right'].set_position(('data', 1.0))
ax[1].spines['top'].set_position(('data', 1.0))

ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('Recall')

# precision-recall curve

ax[2].set_title('Precision-Recall')
ax[2].plot(trained_recall, trained_precision, linestyle='solid', label='Trained', color='xkcd:grass green')
ax[2].plot(balanced_recall, balanced_precision, linestyle='solid', label='Balanced', color='xkcd:blood orange')

# shade the area under the curve
ax[2].fill_between(trained_recall, trained_precision, 0, color='xkcd:light grey green')
ax[2].fill_between(balanced_recall, balanced_precision, 0, color='xkcd:pale salmon')

# adjust the axes to intersect at (0,0) and (1,1)
ax[2].spines['left'].set_position(('data', 0.0))
ax[2].axes.spines['bottom'].set_position(('data', 0.0))
ax[2].axes.spines['right'].set_position(('data', 1.0))
ax[2].axes.spines['top'].set_position(('data', 1.0))

# axis labels
ax[2].set_xlabel('Recall')
ax[2].set_ylabel('Precision')

# show the legend
ax[2].legend(loc=(0.75, 0.85))

# COMMAND ----------

# MAGIC %md この曲線は、バランスのとれたクラスウェイトを適用した場合に何が起こるかを非常に明確に示しています。少数派のクラスを多数派のクラスと同等にすることで、正のクラスの予測精度（precision）は、閾値の増加に伴い、より緩やかに増加しています。 これは、モデルが「安全な」ポジティブを予測するだけでなく、より幅広いポジティブクラスを予測していることを示しています。ポジティブクラスの予測に関してモデルがより積極的になることで、リコールチャートに示されているように、より多くのポジティブクラスのインスタンスが識別されることになります。
# MAGIC 
# MAGIC また、クラスバランシングに伴う精度の低下により、PRCの下の部分が低くなっています。 しかし、このモデルは、以前よりも信頼性の高い解約予測モデルになると思われます。
# MAGIC 
# MAGIC クラスの重み付けに関して最後に注意すべきことは、与えられたクラスのすべてのメンバーに1つの重みを適用する必要はないということです。 その代わり、サンプル加重と呼ばれる手法で、インスタンスごとに加重を割り当てることができます。 このノートには掲載されていませんが、このような手法を使えば、CLVなどの指標に基づいて、ポジティブクラスの異なるインスタンスを評価することができます。 このようなアプローチにより、最大の利益を維持するためにモデルをトレーニングすることができます。技術の進化](https://www.sciencedirect.com/science/article/abs/pii/S0377221718310166)に伴い、これはこの演習の再検討に値する側面かもしれません。

# COMMAND ----------

# MAGIC %md ###ステップ4： 様々なアルゴリズムを評価する
# MAGIC 
# MAGIC クラスの不均衡に対処するためのもう一つの提案は、特定のデータセットとの組み合わせで、あるアルゴリズムが他よりも優れた性能を発揮するかどうかを調べることです。 先ほど、ロジスティック回帰アルゴリズムを使用しましたが、ランダムフォレスト、グラディエントブーステッドツリー、ニューラルネットワークは、これらのシナリオで良い結果をもたらすことがわかっています。それぞれのアルゴリズムがデータセットに対してどのように振る舞うかを見るために、ほぼデフォルトのパラメータ設定でそれぞれのインスタンスを学習させ、それらが*すぐに*どのように動作するかを確認します。 これは、各モデルを網羅的に評価するものではありませんが、理想的には、私たちのデータに適した1つまたは複数のモデルタイプを見つけることができます。

# COMMAND ----------

# DBTITLE 1,Logistic Regression
# train the model
lreg_model = LogisticRegression(max_iter=10000, class_weight='balanced')
lreg_model.fit(X_train, y_train)

# predict
lreg_y_prob = lreg_model.predict_proba(X_test)

# evaluate
lreg_ap = average_precision_score(y_test, lreg_y_prob[:,1])

# COMMAND ----------

# DBTITLE 1,Random Forest
# train the model
rfc_model = RandomForestClassifier(class_weight='balanced')
rfc_model.fit(X_train, y_train)

# predict
rfc_y_prob = rfc_model.predict_proba(X_test)

# evaluate
rfc_ap = average_precision_score(y_test, rfc_y_prob[:,1])

# COMMAND ----------

# DBTITLE 1,Extreme Gradient Boosted Tree (XGBoost)
# normalize class weights so that positive class reflects a 1.0 weight on negative class
scale = weights[1]/weights[0]

# train the model
xgb_model = XGBClassifier(scale_pos_weight=scale) # similar to class_weights arg but applies to positive class only
xgb_model.fit(X_train, y_train)

# predict
xgb_y_prob = xgb_model.predict_proba(X_test)

# evaluate
xgb_ap = average_precision_score(y_test, xgb_y_prob[:,1])

# COMMAND ----------

# MAGIC %md **注意** MLP Classifierはクラスやサンプルの重み付けをサポートしていません。

# COMMAND ----------

# DBTITLE 1,Neural Network
# train the model
mlp_model = MLPClassifier(activation='relu', max_iter=1000)  # does not support class weighting
mlp_model.fit(X_train, y_train)

# predict
mlp_y_prob = mlp_model.predict_proba(X_test)

# evaluate
mlp_ap = average_precision_score(y_test, mlp_y_prob[:,1])

# COMMAND ----------

# MAGIC %md それでは、各モデルの評価指標を比較してみましょう。

# COMMAND ----------

# DBTITLE 1,モデル結果の比較
print('Logistic Regression AP:\t\t{0:.6f}'.format(lreg_ap))
print('RandomForest Classifier AP:\t{0:.6f}'.format(rfc_ap))
print('XGBoost Classifier AP:\t\t{0:.6f}'.format(xgb_ap))
print('MLP (Neural Network) AP:\t{0:.6f}'.format(mlp_ap))

# COMMAND ----------

# MAGIC %md モデルの中では、XGBClassifier が最も優れた性能を示しました（僅差でニューラルネットワークが続きました）。これは、XGBoost が最近の多くのデータ分類コンテストで大きく取り上げられており、このデータセットに見られるような「クラスの不均衡に比較的弱い」(https://www.sciencedirect.com/science/article/pii/S095741741101342X) と認識されていることを考えると、それほど驚くことではありません。
# MAGIC 
# MAGIC 上記で使用したXGBoost分類器は、数ある勾配ブースト分類器の一つです。 LightGBMはこれらのモデルタイプの中でも特に人気があり、sklearnはその機能を模倣したHistGradientBoostingClassifierを提供しています。
# MAGIC 
# MAGIC **注意** HistGradientBoostingClassifierはクラスの重みをサポートしていないので、同じ効果を得るために重みを割り当てられたセットの各インスタンスでサンプルの重みを使用します。

# COMMAND ----------

# DBTITLE 1,Hist Gradient Boost Classifier
# compute sample weights (functionally equivalent to class weights when done in this manner)
sample_weights = compute_sample_weight(
  'balanced', 
  y=y_train
  )

# train the model
hgb_model = HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000)
hgb_model.fit(X_train, y_train, sample_weight=sample_weights)  # weighting applied to individual samples

# predict
hgb_y_prob = hgb_model.predict_proba(X_test)

# evaluate
hgb_ap = average_precision_score(y_test, hgb_y_prob[:,1])
print('HistGB Classifier AP:\t{0:.6f}'.format(hgb_ap))

# COMMAND ----------

# MAGIC %md HistGradientBoostingClassifierは非常に良い結果となりました。 しかし、ここで行った限られた評価では、どのモデルが他のモデルよりも本当に優れているとは言えません。 むしろ、各モデルをチューニングして最適な予測を行い、モデルの比較を行った上で、どのモデルも検討対象から外すべきだと思います。 しかし、時間が限られているので、ランダムフォレストとロジスティック回帰は、このデータセットでの限られたテスト（および文献の情報）によると、最良の結果を得られない可能性が高いため、今後の検討対象から除外します。
# MAGIC 
# MAGIC 最後に検討すべきモデルは、投票型分類器です。 このモデルは、複数のモデルの予測を組み合わせて、アンサンブル予測を行います。ソフトな投票設定では、モデルに提供された各モデルによって生成された確率を平均化するように指示します。 いくつかのモデルが他のモデルよりも信頼性が高い場合は、投票の計算に重み付けを適用して、それらのモデルをより高く評価することができます。 次のノートでは、モデルの重み付けについて説明しますが、ここでは、3つのモデルを同じ重み付けで組み合わせることで、評価指標にどのような影響があるかを見てみましょう。

# COMMAND ----------

# DBTITLE 1,Voting Ensemble
# train the model
vote_model = VotingClassifier(
  estimators=[
    ('hgb', HistGradientBoostingClassifier(loss='binary_crossentropy', max_iter=1000)), 
    ('xgb', XGBClassifier()),
    ('mlp', MLPClassifier(activation='relu', max_iter=1000))
    ],
  voting='soft'
  )
vote_model.fit(X_train, y_train)

# predict
vote_y_prob = vote_model.predict_proba(X_test)

# evaluate
vote_ap = average_precision_score(y_test, vote_y_prob[:,1])
print('Voting AP:\t{0:.6f}'.format(vote_ap))

# COMMAND ----------

# MAGIC %md これらのモデルを一緒に使うと、それぞれのモデルよりも少し良いパフォーマンスが得られます。個々のモデルをチューニングした後（次のノートブックで）、それらを組み合わせるためにどのように投票アンサンブルを使用するかを再検討します。

# COMMAND ----------

# MAGIC %md ###ステップ5：追加オプションの検討
# MAGIC 
# MAGIC クラスの不均衡は、信頼性の高い分類モデルを生成する上で、特に厄介な問題です。インバランスに対処するための多くの追加戦略が確認されており、以下のカテゴリーに分類される傾向があります： </p> <p>1.
# MAGIC 1. アルゴリズムの変更
# MAGIC 2. データセットの変更
# MAGIC 
# MAGIC アルゴリズムの変更」のカテゴリでは、(上記のように) クラスの不均衡に対する感度が低いモデルの異なるクラスを検討することができます。 また、繰り返し最適化を行う際に、異なるアルゴリズムが内部的に使用するペナルティ（クラスの重み）を調整して、少数派のポジティブなクラスをより考慮する機会を探すこともできます。ここでは、これらのテクニックを簡単に紹介しましたが、次のノートブックでは、3つのトップパフォーマンスモデルと投票アンサンブルを使って、引き続き作業を行います。
# MAGIC 
# MAGIC データセットの変更」のカテゴリーでは、少数派クラスのオーバーサンプリング、多数派クラスのアンダーサンプリング、またはこの2つのテクニックの組み合わせが、モデルをより信頼性の高い予測へと導くのに役立つことが示されています。 [これらの手法](https://imbalanced-learn.readthedocs.io/en/stable/introduction.html)は、データセットからランダムに値を選択したり、ML-guidedアプローチを使って選択する値を特定したり、さらにはML-guidedテクニックを使って、マイノリティクラスの値の新しいインスタンスを合成したり、マジョリティクラスの値を*修正したりします。
# MAGIC 
# MAGIC とはいえ、私たちのデータセットにおける否定的なクラスと肯定的なクラスの比率は、およそ30：1です。データセットを修正する」というカテゴリーの技術のほとんどは、クラス比が100:1以上の、より高度に不均衡なシナリオを対象としています。いくつかの限定的なテスト（ここでは示していません）を行った結果、重み付けも代替のサンプリング技術も、評価指標を大きく改善しないことがわかりました。しかし、30:1以下のクラスの不均衡では、このような技術では改善できないというわけではありません。 これらの技術をめぐる文献の一貫したテーマは、1つの技術ですべてのインバランス問題を解決できるわけではなく、結果はデータセットによって大きく異なるということです。
