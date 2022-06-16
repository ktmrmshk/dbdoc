# Databricks notebook source
# MAGIC %md このノートブックの目的は、クラスタリング技術を使って我々の世帯の潜在的なセグメントを特定することである。このノートブックはDatabricks ML 8.0 CPUベースのクラスタで開発されました。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

import numpy as np
import pandas as pd

import mlflow
import os

from delta.tables import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
import seaborn as sns

# COMMAND ----------

# MAGIC %md ## ステップ1：特徴の取得
# MAGIC 
# MAGIC 前回のノートブックで行った作業により、元の特徴セットで見つかったバリエーションを捉えた、限られた数の特徴によって家庭が識別されるようになりました。 これらの特徴は、以下のようにして取得することができます。

# COMMAND ----------

# DBTITLE 1,変換後の特徴量を取得する
# retrieve household (transformed) features
household_X_pd = spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`').toPandas()

# remove household ids from dataframe
X = household_X_pd.drop(['household_id'], axis=1)

household_X_pd

# COMMAND ----------

# MAGIC %md 各特徴の正確な意味を明確にすることは、その工学的な複雑な変換を考えると非常に困難である。 それでも、クラスタリングに利用することは可能です。 (次のノートブックで行うプロファイリングにより、各クラスタの性質を知ることができます)。
# MAGIC 
# MAGIC 最初のステップとして、データを可視化し、自然なグループ分けがあるかどうかを見てみましょう。 超次元空間を扱うので、データを完全に可視化することはできませんが、2次元表示（最初の2つの主成分特徴を使用）により、データに大きなサイズのクラスターがあること、さらにいくつかの緩やかに組織化されたクラスターがあることが分かります。

# COMMAND ----------

# DBTITLE 1,世帯をプロット
fig, ax = plt.subplots(figsize=(10,8))

_ = sns.scatterplot(
  data=X,
  x='Dim_1',
  y='Dim_2',
  alpha=0.5,
  ax=ax
  )

# COMMAND ----------

# MAGIC %md ## ステップ2：K-Meansクラスタリング
# MAGIC 
# MAGIC クラスタリングの最初の試みは、K-means アルゴリズムを利用することです。K-meansは、あらかじめ定義された数の*セントロイド*（クラスタ中心）の周りのクラスタにインスタンスを分割するためのシンプルで一般的なアルゴリズムである。 このアルゴリズムは、クラスタ・センターとして機能する空間内の点の初期セットを生成することで機能します。 次に、インスタンスはこれらの点のうち最も近いものと関連付けられてクラスターを形成し、結果として得られるクラスターの真の中心が再計算される。 そして、新しい中心を使用してクラスタ・メンバーを再登録し、安定した解が生成されるまで（または最大反復回数を使い切るまで）このプロセスが繰り返される。このアルゴリズムの簡単なデモ実行では、次のような結果が得られます。

# COMMAND ----------

# DBTITLE 1,クラスターアサインメントを実証
# initial cluster count
initial_n = 4

# train the model
initial_model = KMeans(
  n_clusters=initial_n,
  max_iter=1000
  )

# fit and predict per-household cluster assignment
init_clusters = initial_model.fit_predict(X)

# combine households with cluster assignments
labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(init_clusters,columns=['cluster'])],
    axis=1
    )
  )

# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / initial_n) for i in range(initial_n)],
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md この最初のモデル実行は、K-meansクラスタリングソリューションの生成の仕組みを示すものですが、このアプローチの欠点もいくつか示しています。 まず、クラスタ数を指定する必要があります。 この値を間違って設定すると、多数の小さなクラスタ、あるいは少数の大きなクラスタが作成されることになり、どちらも、データに内在する、より直接的で自然な構造を観察することを反映しない可能性がある。
# MAGIC 
# MAGIC 第二に、このアルゴリズムの結果は、初期化されたセントロイドに大きく依存する。K-means++の初期化アルゴリズムの使用は、初期セントロイドが人口空間全体に分散していることをより確実にすることで、これらの問題のいくつかを解決する。しかし、これらの選択にはまだランダム性の要素があり、我々の結果に大きな影響を与える可能性がある。
# MAGIC 
# MAGIC これらの課題を解決するために、潜在的なクラスタ数の範囲において、多数のモデル実行を生成します。各実行において、メンバー間の二乗距離の合計と割り当てられたクラスタ中心点（*イナーシャ*）、およびクラスタ間の凝集とクラスタ内の分離を組み合わせた指標（-1～1の範囲、値が高いほど良好）を提供する二次指標（*シルエットスコア*）を算出する。反復処理の回数が多いため、この作業をDatabricksクラスタに分散させ、タイムリーに終了させる予定です。
# MAGIC 
# MAGIC **注** Spark RDDは、パラメータ空間を分散して網羅的に探索する粗い方法として使用しています。これは、定義された値の範囲を効率的に検索するために頻繁に使用される簡単なテクニックです。

# COMMAND ----------

# DBTITLE 1,Iterate over Potential Values of K
# broadcast features so that workers can access efficiently
X_broadcast = sc.broadcast(X)

# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))


# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(sc.defaultParallelism)
    .select('n')
    .rdd
    )

# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )

# remove broadcast set from workers
X_broadcast.unpersist()

display(results_pd)

# COMMAND ----------

# MAGIC %md イナーシャをn、つまり目標とするクラスタ数に対してプロットすると、クラスタメンバーとクラスタセンターの間の二乗距離の総和が、ソリューションのクラスタ数を増やすにつれて減少していることがわかります。 我々の目標は、慣性をゼロにすることではなく（これは、各メンバーをそれ自身の1メンバー・クラスターの中心にした場合に達成されます）、慣性の漸進的な低下が減少する曲線のポイントを特定することです。 このプロットでは、2クラスタと6クラスタの間のどこかにこのポイントがあると見なすことができる。

# COMMAND ----------

# DBTITLE 1,Inertia over Cluster Count
display(results_pd)

# COMMAND ----------

# MAGIC %md イナーシャの*elbow chart*/*scree plot* の解釈はかなり主観的であり、そのため、別のメトリックがクラスタ数に対してどのように振る舞うかを調べることが有用である。 シルエットスコアをnに対してプロットすることで、スコアが低下するピーク（*knee*）を特定する機会を得ることができます。 特に、イナーシャスコアよりもシルエットスコアの方が変化が大きいので、そのピークの位置を正確に特定することが課題です。

# COMMAND ----------

# DBTITLE 1,Silhouette Score over Cluster Count
display(results_pd)

# COMMAND ----------

# MAGIC %md 第2の視点を提供する一方で、シルエットスコアのプロットは、K-meansのクラスタ数の選択は少し主観的であるという概念を補強します。 ドメインの知識とこれらと同様のチャート（[ギャップ統計](https://towardsdatascience.com/k-means-clustering-and-the-gap-statistics-4c5d414acd29)のチャートなど）からの入力が最適なクラスタ数の方向を示すのに役立つかもしれませんが、この値を決定する広く受け入れられた客観的な手段は今のところありません。
# MAGIC 
# MAGIC **NOTE** ニーチャートでシルエットスコアの最高値を追い求めないように注意する必要があります。より高いスコアは、外れ値を単純に小さなクラスタに押し込むことによって、より高いnの値で得ることができる。
# MAGIC 
# MAGIC 慣性のプロットを見ると、この値を支持する証拠があるように見える。 シルエットスコアを調べると、この値では、より低い範囲の値よりも、クラスタリング解がより安定しているように見えます。また、この演習で得られたクラスタの数は、実用的な数である可能性があります。 しかし、最も重要なことは、私たちの可視化から、よく分離された2つのクラスタの存在が自然に目に飛び込んでくることです。
# MAGIC 
# MAGIC nの値が決まったら、次は最終的なクラスターデザインを作成する必要があります。 K-meansの実行から得られる結果のランダム性（大きく変化するシルエット・スコアで捕らえられたように）を考えると、クラスタ・モデルを定義するために*best-of-k*アプローチを取ることができます。 このようなアプローチでは、いくつかのK-meansモデルの実行を行い、シルエット・スコアのような指標によって測定される最良の結果をもたらす実行を選択します。この作業を分散させるために、カスタム関数を実装し、各ワーカーにベスト・オブ・K解を求める作業をさせ、その結果から全体のベスト解を求めるようにします。
# MAGIC 
# MAGIC **注** ここでもRDDを使用して、作業をクラスタ全体に分散できるようにしています。 *iterations* RDDは、実行する各反復のための値を保持します。 *mapPartitions()* を使って、与えられたパーティションに割り当てられた反復処理の数を決定し、そのワーカーに適切に構成されたベストオブK評価を実行させます。 各パーティションは発見できた最良のモデルを返送し、その中から最良のものを選びます。

# COMMAND ----------

# DBTITLE 1,Identify Best of K Model
total_iterations = 50000
n_for_bestofk = 2 
X_broadcast = sc.broadcast(X)

def find_bestofk_for_partition(partition):
   
  # count iterations in this partition
  n_init = sum(1 for i in partition)
  
  # perform iterations to get best of k
  model = KMeans( n_clusters=n_for_bestofk, n_init=n_init, init='k-means++', max_iter=10000)
  model.fit(X_broadcast.value)
  
  # score model
  score = float(silhouette_score(X_broadcast.value, model.labels_))
  
  # return (score, model)
  yield (score, model)


# build RDD for distributed iteration
iterations = sc.range(
              total_iterations, 
              numSlices= sc.defaultParallelism * 4
              ) # distribute work into fairly even number of partitions that allow us to track progress
                        
# retreive best of distributed iterations
bestofk_results = (
  iterations
    .mapPartitions(find_bestofk_for_partition)
    .sortByKey(ascending=False)
    .take(1)
    )[0]

# get score and model
bestofk_score = bestofk_results[0]
bestofk_model = bestofk_results[1]
bestofk_clusters = bestofk_model.labels_

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(bestofk_score))

# combine households with cluster assignments
bestofk_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(bestofk_clusters,columns=['cluster'])],
    axis=1
    )
  )
                        
# clean up 
X_broadcast.unpersist()

# COMMAND ----------

# MAGIC %md この結果を視覚化することで、クラスタとデータの構造がどのように整合しているかを知ることができます。

# COMMAND ----------

# DBTITLE 1,Visualize Best of K Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=bestofk_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_bestofk) for i in range(n_for_bestofk)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md この分析結果は画期的なものではありませんが、そうである必要はないでしょう。 このデータから、これらの特徴を持つ顧客世帯は、2つのグループに分かれて存在すると考えるのが妥当でしょう。 しかし、個々の世帯がどの程度これらのグループに属しているのか、インスタンスごとのシルエットチャートで確認することができます。
# MAGIC 
# MAGIC **注** このコードは、Sci-Kit Learnのドキュメントで提供されている[silhouette charts](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)を修正したものを表しています。

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def plot_silhouette_chart(features, labels):
  
  n = len(np.unique(labels))
  
  # configure plot area
  fig, ax = plt.subplots(1, 1)
  fig.set_size_inches(8, 5)

  # configure plots for silhouette scores between -1 and 1
  ax.set_xlim([-0.1, 1])
  ax.set_ylim([0, len(features) + (n + 1) * 10])
  
  # avg silhouette score
  score = silhouette_score(features, labels)

  # compute the silhouette scores for each sample
  sample_silhouette_values = silhouette_samples(features, labels)

  y_lower = 10

  for i in range(n):

      # get and sort members by cluster and score
      ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
      ith_cluster_silhouette_values.sort()

      # size y based on sample count
      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      # pretty up the charts
      color = cm.nipy_spectral(float(i) / n)
      
      ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # label the silhouette plots with their cluster numbers at the middle
      ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples


  ax.set_title("Average silhouette of {0:.3f} with {1} clusters".format(score, n))
  ax.set_xlabel("The silhouette coefficient values")
  ax.set_ylabel("Cluster label")

  # vertical line for average silhouette score of all the values
  ax.axvline(x=score, color="red", linestyle="--")

  ax.set_yticks([])  # clear the yaxis labels / ticks
  ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
  
  return fig, ax

_ = plot_silhouette_chart(X, bestofk_clusters)

# COMMAND ----------

# MAGIC %md シルエットチャートから、1つのクラスタが他より少し大きく見える。 そのクラスターは、それなりにまとまりがあるように見えます。 他のクラスターは、もう少し分散しているように見え、シルエットスコアの値がより急速に減少し、最終的に数人のメンバーがマイナスのシルエットスコアを持つ（他のクラスターとの重複を示す）ようになります。
# MAGIC 
# MAGIC この解決策は、プロモーションオファーに関連する顧客の行動をより良く理解するために有用であると思われます。他のクラスタリング手法を検討する前に、クラスタの割り当てを持続させる。

# COMMAND ----------

# DBTITLE 1,結果の永続化
# persist household id and cluster assignment
( 
  spark # bring together household and cluster ids
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(bestofk_clusters,columns=['bestofk_cluster'])],
          axis=1
          )[['household_id','bestofk_cluster']]   
      )
    .write  # write data to delta 
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .save('/tmp/completejourney/gold/household_clusters/')
  )

# COMMAND ----------

# MAGIC %md ## ステップ 3: 階層的クラスタリング
# MAGIC 
# MAGIC K-meansに加えて、階層的クラスタリング技術が顧客セグメンテーションの演習で頻繁に使用される。これらの技術の凝集型バリエーションでは、クラスタは、互いに最も近いメンバーをリンクすることによって形成され、セットのすべてのメンバーを包含する単一のクラスタが形成されるまで、それらのクラスタをリンクしてより高いレベルのクラスタを形成する。
# MAGIC 
# MAGIC K-meansとは異なり、凝集過程は決定論的であり、同じデータセットで繰り返し実行すると同じクラスタリング結果が得られる。したがって、階層型クラスタリング技術はK-meansよりも遅いという批判をよく受けるが、ベストオブ*な結果を得るためにアルゴリズムを繰り返し実行する必要がないため、特定の結果を得るための全体的な処理時間は短縮される可能性がある。
# MAGIC 
# MAGIC この技術がどのように機能するかをより良く理解するために、階層型クラスタリング解を学習させ、その出力を可視化してみましょう。

# COMMAND ----------

# DBTITLE 1,デンドログラムを作成する関数
# modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

# function to generate dendrogram
def plot_dendrogram(model, **kwargs):

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
                      [model.children_, 
                       model.distances_,
                       counts]
                      ).astype(float)

    # Plot the corresponding dendrogram
    j = 5
    set_link_color_palette(
      [matplotlib.colors.rgb2hex(cm.nipy_spectral(float(i) / j)) for i in range(j)]
      )
    dendrogram(linkage_matrix, **kwargs)

# COMMAND ----------

# DBTITLE 1,階層的なモデルの学習と可視化
# train cluster model
inithc_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='ward')
inithc_model.fit(X)

# generate visualization
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(15, 8)

plot_dendrogram(inithc_model, truncate_mode='level', p=6) # 6 levels max
plt.title('Hierarchical Clustering Dendrogram')
_ = plt.xlabel('Number of points in node (or index of point if no parenthesis)')

# COMMAND ----------

# MAGIC %md デンドログラムは、下から上に読み込まれます。 各初期点は、ある数のメンバーからなるクラスタを表します。 それらのメンバーが集まって特定のクラスタを形成するプロセス全体は可視化されません（ただし、プロセスのさらに下を見るために *plot_dendrograms* 関数の *p* 引数を調整することができます）。
# MAGIC 
# MAGIC デンドログラムの上部に移動すると、クラスターは新しいクラスターを形成するために収束します。 その収束点に到達するために横断した垂直方向の長さは、これらのクラスター間の距離について何かを教えてくれます。 長さが長いほど、収束するクラスター間のギャップが広くなります。
# MAGIC 
# MAGIC デンドログラムは、データセットの全体的な構造がどのように組み合わされているかを教えてくれますが、最終的なクラスタリングソリューションのための特定のクラスタ数の方向性を示してはくれません。 そのため、我々のソリューションに適切なクラスタ数を特定するために、シルエットスコアのようなメトリックのプロットに戻る必要があります。
# MAGIC 
# MAGIC 様々なクラスタ数に対してシルエットをプロットする前に、クラスタが結合されて新しいクラスタを形成する手段を調べることが重要です。 これには多くのアルゴリズム（*リンク*）がある． SciKit-Learnライブラリは、そのうちの4つをサポートしています。 これらは
# MAGIC <p>
# MAGIC   
# MAGIC * 新しいクラスタ内の二乗距離の合計が最小になるように、クラスタをリンクします。
# MAGIC * 平均* - クラスタ内のすべての点間の平均距離に基づいてクラスタをリンクします。
# MAGIC * シングル* - クラスタ内の任意の2点間の最小距離に基づいてクラスタをリンクします。
# MAGIC * compplete* - クラスタ内の任意の2点間の最大距離に基づいてクラスタをリンクします。
# MAGIC   
# MAGIC リンクのメカニズムが異なると、クラスタリングの結果が大きく異なることがあります。Wardの方法（*ward*リンク・メカニズムで示される）は、ドメインの知識によって別の方法を使用する必要がない限り、ほとんどのクラスタリング演習で使用されると考えられています。

# COMMAND ----------

# DBTITLE 1,クラスター数の特定
results = []

# train models with n number of clusters * linkages
for a in ['ward']:  # linkages
  for n in range(2,21): # evaluate 2 to 20 clusters

    # fit the algorithm with n clusters
    model = AgglomerativeClustering(n_clusters=n, linkage=a)
    clusters = model.fit(X).labels_

    # capture the inertia & silhouette scores for this value of n
    results += [ (n, a, silhouette_score(X, clusters)) ]

results_pd = pd.DataFrame(results, columns=['n', 'linkage', 'silhouette'])
display(results_pd)

# COMMAND ----------

# MAGIC %md その結果、5クラスタを使用した場合に、最良の結果が得られる可能性があることがわかりました。

# COMMAND ----------

# DBTITLE 1,Train & Evaluate Model
n_for_besthc = 5
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# MAGIC %md これらのクラスターを視覚化することで、データ構造の中にどのようにグループ化が存在するのかを見ることができます。 最初の特徴の可視化では、目立つ2つのハイレベルなクラスタがあると主張しました（そして、K-meansアルゴリズムはこれを非常にうまくピックアップしているように見えました）。 しかし、階層型クラスタリング・アルゴリズムでは、より緩やかなサブクラスタが若干検出されたようです。ただし、非常に小さなクラスタについては、緩やかに組織化された世帯が検出されたようです。

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# MAGIC %md インスタンスごとのシルエット・スコアを見ると、このレベルで調べた場合、クラスタ間にもう少し重複があることがわかります。 特に、2次元の可視化を確認すると、これらのポイントは他のクラスターと非常に混在しているように見えます（少なくともこの観点から見た場合）。

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# MAGIC %md それを踏まえて、クラスタ数を4にしてモデルを再トレーニングし、その結果を永続化します。

# COMMAND ----------

# DBTITLE 1,ReTrain & Evaluate Model
n_for_besthc = 4
linkage_for_besthc = 'ward'
 
# configure model
besthc_model = AgglomerativeClustering( n_clusters=n_for_besthc, linkage=linkage_for_besthc)

# train and predict clusters
besthc_clusters = besthc_model.fit(X).labels_

# score results
besthc_score = silhouette_score(X, besthc_clusters)

# print best score obtained
print('Silhouette Score: {0:.6f}'.format(besthc_score))

# combine households with cluster assignments
besthc_labeled_X_pd = (
  pd.concat( 
    [X, pd.DataFrame(besthc_clusters,columns=['cluster'])],
    axis=1
    )
  )

# COMMAND ----------

# DBTITLE 1,Visualize Clusters
# visualize cluster assignments
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(
  data=besthc_labeled_X_pd,
  x='Dim_1',
  y='Dim_2',
  hue='cluster',
  palette=[cm.nipy_spectral(float(i) / n_for_besthc) for i in range(n_for_besthc)],  # align colors with those used in silhouette plots
  legend='brief',
  alpha=0.5,
  ax = ax
  )
_ = ax.legend(loc='lower right', ncol=1, fancybox=True)

# COMMAND ----------

# DBTITLE 1,Examine Per-Member Silhouette Scores
_ = plot_silhouette_chart(X, besthc_clusters)

# COMMAND ----------

# DBTITLE 1,階層的なクラスタの割り当てを保持するフィールドの追加
# add column to previously created table to allow assignment of cluster ids
# try/except used here in case this statement is being rurun against a table with field already in place
try:
  spark.sql('ALTER TABLE DELTA.`/tmp/completejourney/gold/household_clusters/` ADD COLUMN (hc_cluster integer)')
except:
  pass  

# COMMAND ----------

# DBTITLE 1,階層的なクラスタ割り当てを保持するために、永続化されたデータを更新する。
# assemble household IDs and new cluster IDs
updates = (
  spark
    .createDataFrame(
       pd.concat( 
          [household_X_pd, pd.DataFrame(besthc_clusters,columns=['hc_cluster'])],
          axis=1
          )[['household_id','hc_cluster']]   
      )
  )

# merge new cluster ID data with existing table  
deltaTable = DeltaTable.forPath(spark, '/tmp/completejourney/gold/household_clusters/')

(
  deltaTable.alias('target')
    .merge(
      updates.alias('source'),
      'target.household_id=source.household_id'
      )
    .whenMatchedUpdate(set = { 'hc_cluster' : 'source.hc_cluster' } )
    .execute()
  )

# COMMAND ----------

# MAGIC %md ## ステップ4：その他のテクニック
# MAGIC 
# MAGIC 我々は、利用可能なクラスタリング技術について、表面を削り始めたに過ぎません。 [K-Medoids](https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html) はK-meansのバリエーションで、データセットの実際のメンバーでクラスタを構成します。メンバーの類似性を考慮するための代替方法（ユークリッド距離以外）が可能で、データセットのノイズや外れ値に対してより頑強かもしれません。[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)](https://scikit-learn.org/stable/modules/clustering.html#dbscan) も興味深いクラスタリング手法で、メンバー密度の高い領域でクラスタを特定し、密度の低い領域で分散したメンバーを無視するものです。これは、このデータセットに適した手法と思われますが、DBSCANの検証（図示せず）では、高品質のクラスタリング解を生成するための*epsilon*および*minimum sample count*パラメータ（高密度領域の識別方法を制御）の調整が困難でした。また、[Gaussian Mixture Models](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture-models) は、球形でないクラスターをより簡単に形成することができ、セグメンテーションの演習で人気のある別のアプローチを提供しています。
# MAGIC 
# MAGIC 代替アルゴリズムに加えて、クラスターアンサンブルモデル（別名、*コンセンサスクラスタリング*）の開発における新たな研究があります。[Monti *et al.*](https://link.springer.com/article/10.1023/A:1023949509487)によってゲノム研究への応用のために初めて紹介されたコンセンサスクラスタリングは、幅広いライフサイエンス用途で人気を博しているが、顧客セグメンテーションの分野ではこれまでほとんど採用されていないようである。Pythonでは[OpenEnsembles](https://www.jmlr.org/papers/v19/18-100.html)と[kemlglearn](https://nbviewer.jupyter.org/github/bejar/URLNotebooks/blob/master/Notebooks/12ConsensusClustering.ipynb)というパッケージでコンセンサスクラスタリングをサポートしていますが、[diceR](https://cran.r-project.org/web/packages/diceR/index.html) などのRライブラリでより強固なコンセンサスクラスタリングのサポートが見つかります。これらのパッケージやライブラリを限定的に調査したところ（図示していません）、結果はまちまちでしたが、これはハイパーパラメータのチューニングに関する私たち自身の課題によるところが大きく、アルゴリズム自体にはあまり関係がないのではないかと考えています。
