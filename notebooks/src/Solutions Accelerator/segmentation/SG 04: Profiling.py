# Databricks notebook source
# MAGIC %md  このノートブックの目的は、標準的なプロファイリング技術を活用して、前のノートブックで生成されたクラスタをよりよく理解することです。このノートブックはDatabricks ML 8.0 CPUベースのクラスタ上で開発されました。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
import mlflow

import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md ## ステップ1: セグメンテーションされたデータセットのアセンブル
# MAGIC 
# MAGIC クラスタができましたが、それらが一体何を表しているのかがよくわかりません。 無効または不適切な解を導くかもしれないデータの問題を回避するために行った特徴工学の作業は、データを非常に解釈しにくくしています。 
# MAGIC 
# MAGIC この問題を解決するために、（各世帯に割り当てられた）クラスタラベルと、それぞれに関連する元の特徴を取得します。

# COMMAND ----------

# DBTITLE 1,特徴量・ラベルの取得
# retrieve features and labels
household_basefeatures = spark.table('journey.household_features')
household_finalfeatures = spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`')
labels = spark.table('DELTA.`/tmp/completejourney/gold/household_clusters/`')

# assemble labeled feature sets
labeled_basefeatures_pd = (
  labels
    .join(household_basefeatures, on='household_id')
  ).toPandas()

labeled_finalfeatures_pd = (
  labels
    .join(household_finalfeatures, on='household_id')
  ).toPandas()

# get name of all non-feature columns
label_columns = labels.columns

labeled_basefeatures_pd

# COMMAND ----------

# MAGIC %md これらのデータの分析を進める前に、分析の残りの部分をコントロールするために使用されるいくつかの変数を設定しましょう。 複数のクラスタデザインがありますが、このノートでは、階層型クラスタリングモデルからの結果に注目します。

# COMMAND ----------

# DBTITLE 1,解析するクラスターデザインを設定する
cluster_column = 'hc_cluster'
cluster_count = len(np.unique(labeled_finalfeatures_pd[cluster_column]))
cluster_colors = [cm.nipy_spectral(float(i)/cluster_count) for i in range(cluster_count)]

# COMMAND ----------

# MAGIC %md ## ステップ2: セグメントのプロファイル
# MAGIC 
# MAGIC まず始めに、クラスターの2次元の可視化をもう一度見て、クラスターの方向性を確認しましょう。 このグラフで使用している色分けは、残りの可視化にも適用され、探索中のクラスターを容易に判断できるようになります。

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_finalfeatures_pd,
  x='Dim_1',
  y='Dim_2',
  hue=cluster_column,
  palette=cluster_colors,
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md  私たちが考えたセグメントデザインは、同じ大きさのグループを作るのではありません。 その代わり、1つのグループは他のグループより少し大きくなりますが、小さなグループもチームにとって有用な大きさです。

# COMMAND ----------

# DBTITLE 1,クラスタあたりのメンバー数を見る
# count members per cluster
cluster_member_counts = labeled_finalfeatures_pd.groupby([cluster_column]).agg({cluster_column:['count']})
cluster_member_counts.columns = cluster_member_counts.columns.droplevel(0)

# plot counts
plt.bar(
  cluster_member_counts.index,
  cluster_member_counts['count'],
  color = cluster_colors,
  tick_label=cluster_member_counts.index
  )

# stretch y-axis
plt.ylim(0,labeled_finalfeatures_pd.shape[0])

# labels
for index, value in zip(cluster_member_counts.index, cluster_member_counts['count']):
    plt.text(index, value, str(value)+'\n', horizontalalignment='center', verticalalignment='baseline')

# COMMAND ----------

# MAGIC %md 次に、各セグメントが基本特徴量と比較してどのように異なるかを検証してみましょう。 カテゴリ特徴については、クラスタ・メンバー全体の数に対する、特定のプロモーション活動に参加していると識別されたクラスタ・メンバーの割合をプロットします。連続特徴量については、ひげプロットを用いて値を可視化します。

# COMMAND ----------

# DBTITLE 1,プロット描画のための関数を定義する
def profile_segments_by_features(data, features_to_plot, label_to_plot, label_count, label_colors):
  
    feature_count = len(features_to_plot)
    
    # configure plot layout
    max_cols = 5
    if feature_count > max_cols:
      column_count = max_cols
    else:
      column_count = feature_count      
      
    row_count = math.ceil(feature_count / column_count)

    fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4, row_count * 4))
    
    # for each feature (enumerated)
    for k in range(feature_count):

      # determine row & col position
      col = k % column_count
      row = int(k / column_count)
      
      # get axis reference (can be 1- or 2-d)
      try:
        k_ax = ax[row,col]
      except:
        pass
        k_ax = ax[col]
      
      # set plot title
      k_ax.set_title(features_to_plot[k].replace('_',' '), fontsize=7)

      # CATEGORICAL FEATURES
      if features_to_plot[k][:4]=='has_': 

        # calculate members associated with 0/1 categorical values
        x = data.groupby([label_to_plot,features_to_plot[k]]).agg({label_to_plot:['count']})
        x.columns = x.columns.droplevel(0)

        # for each cluster
        for c in range(label_count):

          # get count of cluster members
          c_count = x.loc[c,:].sum()[0]

          # calculate members with value 0
          try:
            c_0 = x.loc[c,0]['count']/c_count
          except:
            c_0 = 0

          # calculate members with value 1
          try:
            c_1 = x.loc[c,1]['count']/c_count
          except:
            c_1 = 0

          # render percent stack bar chart with 1s on bottom and 0s on top
          k_ax.set_ylim(0,1)
          k_ax.bar([c], c_1, color=label_colors[c], edgecolor='white')
          k_ax.bar([c], c_0, bottom=c_1, color=label_colors[c], edgecolor='white', alpha=0.25)


      # CONTINUOUS FEATURES
      else:    

        # get subset of data with entries for this feature
        x = data[
              ~np.isnan(data[features_to_plot[k]])
              ][[label_to_plot,features_to_plot[k]]]

        # get values for each cluster
        p = []
        for c in range(label_count):
          p += [x[x[label_to_plot]==c][features_to_plot[k]].values]

        # plot values
        k_ax.set_ylim(0,1)
        bplot = k_ax.boxplot(
            p, 
            labels=range(label_count),
            patch_artist=True
            )

        # adjust box fill to align with cluster
        for patch, color in zip(bplot['boxes'], label_colors):
          patch.set_alpha(0.75)
          patch.set_edgecolor('black')
          patch.set_facecolor(color)
    

# COMMAND ----------

# DBTITLE 1,全特徴量ごとのレンダリングプロット
# get feature names
feature_names = labeled_basefeatures_pd.drop(label_columns, axis=1).columns

# generate plots
profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md このプロットには検討すべきことがたくさんありますが、最も簡単なのは、あるプロモーション・オファーに反応するグループとしないグループを識別するために、カテゴリ的特徴から始めることだと思われます。 連続的な特徴は、そのクラスタが反応したときのエンゲージメントの度合いについて、もう少し詳しく知ることができます。 
# MAGIC 
# MAGIC 様々な機能を使いこなすうちに、異なるクラスタについての説明ができ始めるでしょう。 これを支援するために、特徴の特定のサブセットを取り出し、より少ない数の特徴に注意を集中させることが有効な場合があります。

# COMMAND ----------

# DBTITLE 1,いくつかをピックアップ
feature_names = ['has_pdates_campaign_targeted', 'pdates_campaign_targeted', 'amount_list_with_campaign_targeted']

profile_segments_by_features(labeled_basefeatures_pd, feature_names, cluster_column, cluster_count, cluster_colors)

# COMMAND ----------

# MAGIC %md ## ステップ3：セグメントの記述
# MAGIC 
# MAGIC 特徴をよく調べると、行動の観点からクラスターを区別できるようになることが期待されます。 ここで、これらのグループがなぜ存在するのか、また、複数年の行動情報を収集することなく、どのようにしてグループのメンバーである可能性を特定することができるのかを検討することが興味深くなってきます。これを行う一般的な方法は、クラスター設計で採用されなかった特性からクラスターを調べることです。このデータセットでは、この目的のために、世帯の一部で利用可能な人口統計学的情報を使用することができる。

# COMMAND ----------

# DBTITLE 1,世帯の属性とクラスターラベルの関連付け
labels = spark.table('DELTA.`/tmp/completejourney/gold/household_clusters/`').alias('labels')
demographics = spark.table('journey.households').alias('demographics')

labeled_demos = (
  labels
    .join(demographics, on=expr('labels.household_id=demographics.household_id'), how='leftouter')  # only 801 of 2500 present should match
    .withColumn('matched', expr('demographics.household_id Is Not Null'))
    .drop('household_id')
  ).toPandas()

labeled_demos

# COMMAND ----------

# MAGIC %md 先に進む前に、クラスタ内のメンバーのうち何人がデモグラフィック情報を持っているのかを考える必要があります。

# COMMAND ----------

# DBTITLE 1,人口統計データを持つクラスター構成員の割合の検討
x = labeled_demos.groupby([cluster_column, 'matched']).agg({cluster_column:['count']})
x.columns = x.columns.droplevel(0)

# for each cluster
for c in range(cluster_count):

  # get count of cluster members
  c_count = x.loc[c,:].sum()[0]

  # calculate members with value 0
  try:
    c_0 = x.loc[c,0]['count']/c_count
  except:
    c_0 = 0

  # calculate members with value 1
  try:
    c_1 = x.loc[c,1]['count']/c_count
  except:
    c_1 = 0
  
  # plot counts
  plt.bar([c], c_1, color=cluster_colors[c], edgecolor='white')
  plt.bar([c], c_0, bottom=c_1, color=cluster_colors[c], edgecolor='white', alpha=0.25)
  plt.xticks(range(cluster_count))
  plt.ylim(0,1)

# COMMAND ----------

# MAGIC %md 理想的には、データセットに含まれる全世帯の人口統計データがあるか、少なくとも各クラスターに含まれるメンバーの大規模で一貫した割合の人口統計データがあることが望ましい。 そうでなければ、これらのデータから結論を導き出すことには慎重でなければならない。
# MAGIC 
# MAGIC それでも、テクニックを示すために、この演習を続けるかもしれません。 このことを念頭に置いて、世帯主の年齢層について分割表を作成し、クラスタのメンバーがどのように年齢を中心に並んでいるかを見てみましょう。

# COMMAND ----------

# DBTITLE 1,Demonstrate Contingency Table
age_by_cluster = sm.stats.Table.from_data(labeled_demos[[cluster_column,'age_bracket']])
age_by_cluster.table_orig

# COMMAND ----------

# MAGIC %md そこで、ピアソンのカイ二乗(*&Chi;^2*)検定を適用して、これらの頻度差が統計的に有意であるかどうかを判断することができるかもしれません。 このような検定では、5%以下のp値は、頻度分布が偶然によるものではない（したがって、カテゴリー割り当てに依存する）ことを教えてくれるでしょう。

# COMMAND ----------

# DBTITLE 1,カイ二乗検定
res = age_by_cluster.test_nominal_association()
res.pvalue

# COMMAND ----------

# MAGIC %md その後、各クラスターと人口統計グループの交点に関連するピアソンの残差を調べ、特定の交点がいつこの結論に至ったかを判断することができるだろう。 残差値が2または4以上の交差点は、それぞれ95%または99.9%の確率で予想と異なり、これらはクラスターを区別する人口統計学的特性である可能性が高いです。

# COMMAND ----------

# DBTITLE 1,ピアソン残差の実証実験
age_by_cluster.resid_pearson  # standard normal random variables within -2, 2 with 95% prob and -4,4 at 99.99% prob

# COMMAND ----------

# MAGIC %md もし、このデータから何か意味のあるものを見つけたとしたら、次の課題は、これらの統計的検定に慣れていないチームのメンバーにそれを伝えることでしょう。 これを行うための一般的な方法は、 *[mosaic plot](https://www.datavis.ca/papers/casm/casm.html#tth_sEc3)* またの名を *marimekko plot* です。

# COMMAND ----------

# DBTITLE 1,モザイクプロット
# assemble demographic category labels as key-value pairs (limit to matched values)
demo_labels = np.unique(labeled_demos[labeled_demos['matched']]['age_bracket'])
demo_labels_kv = dict(zip(demo_labels,demo_labels))

# define function to generate cell labels
labelizer = lambda key: demo_labels_kv[key[1]]

# define function to generate cell colors
props = lambda key: {'color': cluster_colors[int(key[0])], 'alpha':0.8}

# generate mosaic plot
fig, rect = mosaic(
  labeled_demos.sort_values('age_bracket', ascending=False),
  [cluster_column,'age_bracket'], 
  horizontal=True, 
  axes_label=True, 
  gap=0.015, 
  properties=props, 
  labelizer=labelizer
  )

# set figure size
_ = fig.set_size_inches((10,8))

# COMMAND ----------

# MAGIC %md 各カテゴリーに関連するメンバーの比例表示と、クラスタの相対的な幅の比例表示により、これらのグループ間の頻度差を要約する良い方法となります。統計解析と組み合わせて、モザイクプロットは、統計的に有意な発見をより簡単に理解するための素晴らしい方法を提供します。

# COMMAND ----------

# MAGIC %md ## ステップ4：次のステップ
# MAGIC 
# MAGIC セグメンテーションは、1回で終了することはほとんどない。むしろ、今回のデータから学んだことで、差別化できない特徴を取り除き、場合によっては他の特徴を加えて、分析を繰り返すかもしれない。さらに、RFMセグメンテーションやCLV分析など、他の分析も行い、ここで検討したセグメンテーションデザインとの関連性を検討することもある。 最終的には、新しいセグメンテーションデザインに到達するかもしれないが、そうでない場合でも、より良いプロモーションキャンペーンを作るのに役立つ洞察を得ることができたといえる。

# COMMAND ----------


