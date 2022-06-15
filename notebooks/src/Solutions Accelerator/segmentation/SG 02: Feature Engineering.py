# Databricks notebook source
# MAGIC %md このノートブックの目的は、特徴工学と次元削減の技術を組み合わせて、セグメンテーション作業に必要な特徴を生成することです。このノートブックはDatabricks ML 8.0 CPUベースのクラスタ上で開発されました。

# COMMAND ----------

# DBTITLE 1,必要なPythonライブラリのインストール
# MAGIC %pip install dython

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
from sklearn.preprocessing import quantile_transform

import dython
import math

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## ステップ1：ベースとなる特徴の導出
# MAGIC 
# MAGIC 様々なプロモーションに対する反応性から顧客世帯をセグメント化することを目的として、まず、各プロモーション項目の単独および組み合わせによる購入日数（ *pdates_see* ）と販売量（ *amount_list_* ）を計算する。 対象となるプロモーションアイテムは以下の通り。
# MAGIC 
# MAGIC キャンペーン対象製品（ *campaign_targeted_* ） ・プライベートブランド製品（ *private_label_* ） ・その他（ *campaign_targeted_* ）。
# MAGIC * プライベートブランド製品（ *private_tica_label_* ）。
# MAGIC * インストアディスカウント商品 ( *instore_discount_* )
# MAGIC * キャンペーンクーポン ( *campaign_coupon_redemption_* )
# MAGIC * Manufacturer-generated coupon redemption ( *manuf_coupon_redemption_* )
# MAGIC 
# MAGIC この結果は、決して網羅的なものではありませんが、分析の出発点として有用なものです。

# COMMAND ----------

# DBTITLE 1,関連するメトリクスの導出
# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS journey.household_metrics;
# MAGIC 
# MAGIC CREATE VIEW journey.household_metrics
# MAGIC AS
# MAGIC   WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM journey.campaigns a
# MAGIC       INNER JOIN journey.campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN journey.coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       ),
# MAGIC     product_spend AS (
# MAGIC       SELECT
# MAGIC         a.household_id,
# MAGIC         a.product_id,
# MAGIC         a.day,
# MAGIC         a.basket_id,
# MAGIC         CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC         CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC         CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC         CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC         a.amount_list,
# MAGIC         a.campaign_coupon_discount,
# MAGIC         a.manuf_coupon_discount,
# MAGIC         a.total_coupon_discount,
# MAGIC         a.instore_discount,
# MAGIC         a.amount_paid  
# MAGIC       FROM journey.transactions_adj a
# MAGIC       INNER JOIN journey.products b
# MAGIC         ON a.product_id=b.product_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC 
# MAGIC     x.household_id,
# MAGIC 
# MAGIC     -- Purchase Date Level Metrics
# MAGIC     COUNT(DISTINCT x.day) as purchase_dates,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL THEN x.day ELSE NULL END) as pdates_campaign_targeted,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 THEN x.day ELSE NULL END) as pdates_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemptions,
# MAGIC     COUNT(DISTINCT CASE WHEN x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN y.product_id IS NOT NULL AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.campaign_coupon_redemption = 1 AND x.private_label = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COUNT(DISTINCT CASE WHEN x.manuf_coupon_redemption = 1 AND x.instore_discount_applied = 1 THEN x.day ELSE NULL END) as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC 
# MAGIC     -- List Amount Metrics
# MAGIC     COALESCE(SUM(x.amount_list),0) as amount_list,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.amount_list),0) as amount_list_with_campaign_targeted,
# MAGIC     COALESCE(SUM(x.private_label * x.amount_list),0) as amount_list_with_private_label,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.amount_list),0) as amount_list_with_campaign_targeted_private_label,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.amount_list),0) as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.amount_list),0) as amount_list_with_manuf_coupon_redemptions,
# MAGIC     COALESCE(SUM(x.instore_discount_applied * x.amount_list),0) as amount_list_with_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(CASE WHEN y.product_id IS NOT NULL THEN 1 ELSE 0 END * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.campaign_coupon_redemption * x.private_label * x.instore_discount_applied * x.amount_list),0) as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC     COALESCE(SUM(x.manuf_coupon_redemption * x.instore_discount_applied * x.amount_list),0) as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC   FROM product_spend x
# MAGIC   LEFT OUTER JOIN targeted_products_by_household y
# MAGIC     ON x.household_id=y.household_id AND x.product_id=y.product_id
# MAGIC   GROUP BY 
# MAGIC     x.household_id;
# MAGIC     
# MAGIC SELECT * FROM journey.household_metrics;

# COMMAND ----------

# MAGIC %md このデータセットに含まれる世帯は、データ提供期間である711日間のうち、最低限の活動量に基づいて選択されているものと思われます。 しかし、世帯によって期間中の購入頻度や総消費額は異なります。 これらの値を世帯間で正規化するために、各指標を、その世帯の購入履歴に関連する購入日の合計またはリスト金額の合計で割ることにします。
# MAGIC 
# MAGIC **注** この次のステップで行うように、総購入日および総消費額に基づいてデータを正規化することは、すべての分析において適切であるとは限りません。

# COMMAND ----------

# DBTITLE 1,メトリクスを特徴量に変換する
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS journey.household_features;
# MAGIC 
# MAGIC CREATE VIEW journey.household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC       household_id,
# MAGIC   
# MAGIC       pdates_campaign_targeted/purchase_dates as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       pdates_campaign_targeted_private_label/purchase_dates as pdates_campaign_targeted_private_label,
# MAGIC       pdates_campaign_coupon_redemptions/purchase_dates as pdates_campaign_coupon_redemptions,
# MAGIC       pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       pdates_manuf_coupon_redemptions/purchase_dates as pdates_manuf_coupon_redemptions,
# MAGIC       pdates_instore_discount_applied/purchase_dates as pdates_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_instore_discount_applied/purchase_dates as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       amount_list_with_campaign_targeted/amount_list as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       amount_list_with_campaign_targeted_private_label/amount_list as amount_list_with_campaign_targeted_private_label,
# MAGIC       amount_list_with_campaign_coupon_redemptions/amount_list as amount_list_with_campaign_coupon_redemptions,
# MAGIC       amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       amount_list_with_manuf_coupon_redemptions/amount_list as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM journey.household_metrics
# MAGIC ORDER BY household_id;
# MAGIC 
# MAGIC SELECT * FROM journey.household_features;

# COMMAND ----------

# MAGIC %md ## ステップ2: 分布を調べる
# MAGIC 
# MAGIC 先に進む前に、採用するクラスタリング手法との互換性を理解するために、特徴をよく調べることは良いアイデアです。一般的には、比較的正規の分布で標準化されたデータが望ましいのですが、すべてのクラスタリングアルゴリズムに対して難しい要件ではありません。
# MAGIC 
# MAGIC データの分布を調べるために、pandasのDataframeにデータを取り込みます。 もしデータ量が多くてpandasが使えない場合は、Spark DataFrameに対して[*sample()*](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample)を使ってランダムにサンプルを取って、分布を調べることもできます。

# COMMAND ----------

# DBTITLE 1,Retrieve Features
# retreive as Spark dataframe
household_features = (
  spark
    .table('journey.household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# collect some basic info on our features
household_features_pd.info()

# COMMAND ----------

# MAGIC %md このデータセットでは、*household_id* フィールドを取得することにしていることに注意してください。 このような一意な識別子は、この後のデータ変換やクラスタリング作業には渡されませんが、その結果を検証するのに役立つ可能性があります。この情報を特徴量で取得することで、特徴量と一意な識別子を2つの別々のpandasデータフレームに分離し、それぞれのインスタンスを共有インデックス値を利用して簡単に再関連付けできるようになりました。

# COMMAND ----------

# DBTITLE 1,世帯IDと機能の分離
# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md それでは、特徴量の構成について検討を開始します。

# COMMAND ----------

# DBTITLE 1,特徴量の統計・概要
features_pd.describe()

# COMMAND ----------

# MAGIC %md 特徴量をざっと見てみると、平均値が非常に低く、ゼロ値が多いことがわかります（複数の分位点でゼロが発生していることでわかる）。 特徴量の分布を詳しく見て、後で困るようなデータ分布の問題がないことを確認する必要があります。

# COMMAND ----------

# DBTITLE 1,Examine Feature Distributions
feature_names = features_pd.columns
feature_count = len(feature_names)

# determine required rows and columns for visualizations
column_count = 5
row_count = math.ceil(feature_count / column_count)

# configure figure layout
fig, ax = plt.subplots(row_count, column_count, figsize =(column_count * 4.5, row_count * 3))

# render distribution of each feature
for k in range(0,feature_count):
  
  # determine row & col position
  col = k % column_count
  row = int(k / column_count)
  
  # set figure at row & col position
  ax[row][col].hist(features_pd[feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[row][col].set_xlim(0,1)   # set x scale 0 to 1
  ax[row][col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[row][col].text(x=0.1, y=features_pd.shape[0]-100, s=feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md ざっと見たところ、多くの特徴量に *ゼロ膨張分布* があることがわかります。 これは、頻度の低い事象の大きさを計測しようとする場合によくある課題です。 
# MAGIC 
# MAGIC ゼロ膨張分布に対処するための様々なテクニックや、ゼロ膨張分布を扱うために設計されたゼロ膨張モデルについての文献は増えてきていますが、ここでは単純にゼロ膨張分布とゼロ膨張分布を分けて考えます。 1つはイベントの発生をバイナリ（カテゴリカル）特徴として捉え、もう1つはイベントが発生したときの大きさを捉える特徴である。
# MAGIC 
# MAGIC **注**）バイナリ特徴量には、識別しやすいように *has_* という接頭辞をつけることにする。あるイベントに関連する購入日がゼロの世帯は、そのイベントの販売金額値もゼロであると予想されます。そのため、イベントに対して1つのバイナリフィーチャーを作成し、関連する購入日と金額のリスト値に対してそれぞれ2つ目のフィーチャーを作成します。

# COMMAND ----------

# DBTITLE 1,ゼロインフレーテッド・ディストリビューション問題に対応するための機能を定義する。
# MAGIC %sql
# MAGIC 
# MAGIC DROP VIEW IF EXISTS journey.household_features;
# MAGIC 
# MAGIC CREATE VIEW journey.household_features 
# MAGIC AS 
# MAGIC 
# MAGIC SELECT
# MAGIC 
# MAGIC       household_id,
# MAGIC       
# MAGIC       -- binary features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted,
# MAGIC       -- CASE WHEN pdates_private_label > 0 THEN 1 ELSE 0 END as has_pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemptions,
# MAGIC       -- CASE WHEN pdates_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_instore_discount_applied,
# MAGIC       -- CASE WHEN pdates_private_label_instore_discount_applied > 0 THEN 1 ELSE 0 END as has_pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN 1 
# MAGIC         ELSE 0 END as has_pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC   
# MAGIC       -- purchase date features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN pdates_campaign_targeted/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted,
# MAGIC       pdates_private_label/purchase_dates as pdates_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN pdates_campaign_targeted_private_label/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN pdates_campaign_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN pdates_campaign_coupon_redemptions_on_private_labels/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN pdates_manuf_coupon_redemptions/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN pdates_campaign_targeted_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_instore_discount_applied,
# MAGIC       pdates_private_label_instore_discount_applied/purchase_dates as pdates_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN pdates_campaign_targeted_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN pdates_manuf_coupon_redemption_instore_discount_applied/purchase_dates 
# MAGIC         ELSE NULL END as pdates_manuf_coupon_redemption_instore_discount_applied,
# MAGIC       
# MAGIC       -- list amount features
# MAGIC       CASE WHEN pdates_campaign_targeted > 0 THEN amount_list_with_campaign_targeted/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted,
# MAGIC       amount_list_with_private_label/amount_list as amount_list_with_private_label,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label > 0 THEN amount_list_with_campaign_targeted_private_label/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions > 0 THEN amount_list_with_campaign_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemptions_on_private_labels > 0 THEN amount_list_with_campaign_coupon_redemptions_on_private_labels/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemptions_on_private_labels,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemptions > 0 THEN amount_list_with_manuf_coupon_redemptions/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemptions,
# MAGIC       amount_list_with_instore_discount_applied/amount_list as amount_list_with_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_instore_discount_applied,
# MAGIC       amount_list_with_private_label_instore_discount_applied/amount_list as amount_list_with_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_targeted_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_targeted_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_targeted_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_instore_discount_applied,
# MAGIC       CASE WHEN pdates_campaign_coupon_redemption_private_label_instore_discount_applied > 0 THEN amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_campaign_coupon_redemption_private_label_instore_discount_applied,
# MAGIC       CASE WHEN pdates_manuf_coupon_redemption_instore_discount_applied > 0 THEN amount_list_with_manuf_coupon_redemption_instore_discount_applied/amount_list 
# MAGIC         ELSE NULL END as amount_list_with_manuf_coupon_redemption_instore_discount_applied
# MAGIC 
# MAGIC FROM journey.household_metrics
# MAGIC ORDER BY household_id;

# COMMAND ----------

# DBTITLE 1,特徴量をPandasで読み込む
# retreive as Spark dataframe
household_features = (
  spark
    .table('journey.household_features')
  )

# retrieve as pandas Dataframe
household_features_pd = household_features.toPandas()

# get household ids from dataframe
households_pd = household_features_pd[['household_id']]

# remove household ids from dataframe
features_pd = household_features_pd.drop(['household_id'], axis=1)

features_pd

# COMMAND ----------

# MAGIC %md 特徴を分離した上で、特徴の分布をもう一度見てみましょう。 まず、新しいバイナリ特徴量から見ていきます。

# COMMAND ----------

# DBTITLE 1,2値化された特徴の分布を調べる
b_feature_names = list(filter(lambda f:f[0:4]==('has_') , features_pd.columns))
b_feature_count = len(b_feature_names)

# determine required rows and columns
b_column_count = 5
b_row_count = math.ceil(b_feature_count / b_column_count)

# configure figure layout
fig, ax = plt.subplots(b_row_count, b_column_count, figsize =(b_column_count * 3.5, b_row_count * 3.5))

# render distribution of each feature
for k in range(0,b_feature_count):
  
  # determine row & col position
  b_col = k % b_column_count
  b_row = int(k / b_column_count)
  
  # determine feature to be plotted
  f = b_feature_names[k]
  
  value_counts = features_pd[f].value_counts()

  # render pie chart
  ax[b_row][b_col].pie(
    x = value_counts.values,
    labels = value_counts.index,
    explode = None,
    autopct='%1.1f%%',
    labeldistance=None,
    #pctdistance=0.4,
    frame=True,
    radius=0.48,
    center=(0.5, 0.5)
    )
  
  # clear frame of ticks
  ax[b_row][b_col].set_xticks([])
  ax[b_row][b_col].set_yticks([])
  
  # legend & feature name
  ax[b_row][b_col].legend(bbox_to_anchor=(1.04,1.05),loc='upper left', fontsize=8)
  ax[b_row][b_col].text(1.04,0.8, s=b_feature_names[k].replace('_','\n'), fontsize=8, va='top')

# COMMAND ----------

# MAGIC %md 円グラフから、多くのプロモーション・オファーが実行されていないことがわかります。これは、多くのプロモーション・オファー、特にクーポンに関連したプロモーション・オファーに典型的なものです。個別に見ると、多くのプロモーション・オファーの利用率は低いですが、複数のプロモーション・オファーの利用率を互いに組み合わせて調べると、利用率は、個々のオファーに注目する代わりに、組み合わせのオファーを無視することを検討するレベルまで低下しています。この点については、ゼロ・インフレーション補正を行った継続的な機能に目を向けるため、ここでは保留とします。

# COMMAND ----------

# DBTITLE 1,Examine Distribution of Continuous Features
c_feature_names = list(filter(lambda f:f[0:4]!=('has_') , features_pd.columns))
c_feature_count = len(c_feature_names)

# determine required rows and columns
c_column_count = 5
c_row_count = math.ceil(c_feature_count / c_column_count)

# configure figure layout
fig, ax = plt.subplots(c_row_count, c_column_count, figsize =(c_column_count * 4.5, c_row_count * 3))

# render distribution of each feature
for k in range(0, c_feature_count):
  
  # determine row & col position
  c_col = k % c_column_count
  c_row = int(k / c_column_count)
  
  # determine feature to be plotted
  f = c_feature_names[k]
  
  # set figure at row & col position
  ax[c_row][c_col].hist(features_pd[c_feature_names[k]], rwidth=0.95, bins=10) # histogram
  ax[c_row][c_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[c_row][c_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[c_row][c_col].text(x=0.1, y=features_pd.shape[0]-100, s=c_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md 問題の特徴の多くからゼロが取り除かれたことで、より多くの標準的な分布が得られました。 しかし，これらの分布は非正規分布（ガウス分布ではない）である場合があり，ガウス分布は多くのクラスタリング技術で本当に役に立ちます．
# MAGIC 
# MAGIC これらの分布をより正規化する方法の1つは，Box-Cox変換を適用することです． この変換をこれらの特徴量に適用したところ（図示していません）、多くの分布がここに示したものよりもはるかに正規分布にならないことがわかりました。 そこで、もう少し主張の強い別の変換、[分位点変換](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform)を利用することにします。
# MAGIC 
# MAGIC 分位値変換は与えられた特徴量のデータポイントに関連する累積確率関数を計算します。 これは、ある特徴のデータをソートし、観測された値の範囲内で、ある値のパーセントランクを計算する関数を計算することを言い表したものである。このパーセントランキング関数は、データを正規分布のようなよく知られた分布に対応させるための基礎となるものである。この変換の背後にある[正確な数学](https://www.sciencedirect.com/science/article/abs/pii/S1385725853500125)は、この変換の有用性を観察するために、完全に理解する必要はありません。 もしこれが分位値変換の最初の紹介なら、このテクニックは1950年代からあり、多くの学問分野で多用されていることを知るだけでよいでしょう。

# COMMAND ----------

# DBTITLE 1,連続特徴量に分位値変換を適用する
# access continous features
c_features_pd = features_pd[c_feature_names]

# apply quantile transform
qc_features_pd = pd.DataFrame(
  quantile_transform(c_features_pd, output_distribution='normal', ignore_implicit_zeros=True),
  columns=c_features_pd.columns,
  copy=True
  )

# show transformed data
qc_features_pd

# COMMAND ----------

# DBTITLE 1,分位変換された連続特徴量の分布を調べる
qc_feature_names = qc_features_pd.columns
qc_feature_count = len(qc_feature_names)

# determine required rows and columns
qc_column_count = 5
qc_row_count = math.ceil(qc_feature_count / qc_column_count)

# configure figure layout
fig, ax = plt.subplots(qc_row_count, qc_column_count, figsize =(qc_column_count * 5, qc_row_count * 4))

# render distribution of each feature
for k in range(0,qc_feature_count):
  
  # determine row & col position
  qc_col = k % qc_column_count
  qc_row = int(k / qc_column_count)
  
  # set figure at row & col position
  ax[qc_row][qc_col].hist(qc_features_pd[qc_feature_names[k]], rwidth=0.95, bins=10) # histogram
  #ax[qc_row][qc_col].set_xlim(0,1)   # set x scale 0 to 1
  ax[qc_row][qc_col].set_ylim(0,features_pd.shape[0]) # set y scale 0 to 2500 (household count)
  ax[qc_row][qc_col].text(x=0.1, y=features_pd.shape[0]-100, s=qc_feature_names[k].replace('_','\n'), fontsize=9, va='top')      # feature name in chart

# COMMAND ----------

# MAGIC %md ここで重要なのは、分位数変換が強力であるのと同様に、すべてのデータの問題を魔法のように解決できるわけではない、ということです。 このノートブックの開発では、変換後のデータに二峰性分布があるように見えるいくつかの特徴を確認しました。 これらの特徴は、当初ゼロ膨張分布補正を適用しないことにしていたものです。 このような場合、特徴量の定義に戻り、補正を行い、変換を再実行することで問題が解決されました。とはいえ，分布の左端に位置する小さな世帯群がある場合，すべての変換された分布を修正したわけではありません。 そこで、約250世帯以上の世帯がそのビンに含まれるものだけに対処することにしました。

# COMMAND ----------

# MAGIC %md ## ステップ3: 関係を検証する
# MAGIC 
# MAGIC 連続特徴量が正規分布に揃ったので、連続特徴量から始めて特徴変数間の関係を調べましょう。 標準的な相関を使用すると、非常に多くの関連する特徴があることがわかります。 ここに見られる多重共線性は、対処しなければ、プロモーション反応のある側面を過度に強調し、他の側面を弱めるクラスタリングの原因となります。

# COMMAND ----------

# DBTITLE 1,連続的な特徴の関係を調べる
# generate correlations between features
qc_features_corr = qc_features_pd.corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(qc_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  qc_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md そして、バイナリ特徴間の関係についてはどうでしょうか？ ピアソンの相関(上記のヒートマップで使用)は、カテゴリデータを扱う場合、有効な結果を生成しません。そこで、代わりに[Theilの不確実性係数](https://en.wikipedia.org/wiki/Uncertainty_coefficient)を計算します。これは、あるバイナリ測定の値が、どの程度他の値を予測するかを調べるために設計された指標です。 TheilのUは、変数間に予測値がない0と、完全な予測値がある1の間の範囲に収まります。この指標で本当に面白いのは、それが**非対称**であることで、あるバイナリ測定の示すスコアが他を予測するが、必ずしもその逆ではないことである。 これは、以下のヒートマップのスコアを慎重に検討する必要があり、対角線上の出力が対称であると仮定しないことを意味します。
# MAGIC 
# MAGIC **注** メトリクスの計算を行う*dython*パッケージの主要な作者は、TheilのUと関連するメトリクスについて議論した[素晴らしい記事](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)を持っています。

# COMMAND ----------

# DBTITLE 1,Examine Relationships between Binary Features
# generate heatmap with Theil's U
_ = dython.nominal.associations(
  features_pd[b_feature_names], 
  nominal_columns='all',
# theil_u=True,
  nom_nom_assoc='theil',
  figsize=(10,8),
  cmap='coolwarm',
  vmax=1.0,
  vmin=0.0,
  cbar=False
  )

# COMMAND ----------

# MAGIC %md  連続特徴量と同様に、バイナリ変数の関係にも問題があり、対処する必要があります。 また、連続特徴とカテゴリ特徴の関係はどうでしょうか？
# MAGIC 
# MAGIC 値0のバイナリ素性は、関連する連続素性に対してNULL/NaN値を持ち、連続素性の実数値は関連するバイナリ素性の値1に変換されることが、その由来から分かっています。これらの特徴の間に関係があることを知るために指標を計算する必要はありません（ただし、疑問があれば[Correlation Ratio](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9)を計算することで解決できるかもしれません）。 では、このような特徴量データの関係性に対処するためにはどうすればよいのでしょうか。
# MAGIC 
# MAGIC 多数の特徴量を扱う場合、これらの関係には通常、次元削減技術を用います。この手法は、データの変動の大部分がより少ない数の特徴によって捉えられるように、データを投影するものである。 これらの特徴は、潜在因子または主成分と呼ばれることが多く（採用する手法によって異なる）、表面レベルの特徴に反映されているデータの基本構造を捕らえ、特徴の重複する説明力、すなわち多重共線性を除去する方法でこれを行う。
# MAGIC 
# MAGIC では、どの次元削減手法を使えばよいのでしょうか？ **主成分分析（PCA）** は、これらの手法の中で最も一般的ですが、連続特徴からなるデータセットにのみ適用できます。**混合成分分析(MCA)** もこれらの手法の1つですが、カテゴリ的な特徴を持つデータセットにのみ適用できます。**混合データの因子分析（FAMD）** は、連続データとカテゴリデータの両方からなるデータに対して、これら2つの手法の概念を組み合わせて、削減された特徴セットを構築することができます。 しかし、FAMDを我々の特徴量データに適用するには問題がある。
# MAGIC 
# MAGIC PCAとMCA（したがってFAMD）の典型的な実装では、データ中に欠損値が存在しないことが要求されます。 連続特徴量の場合は平均値や中央値、カテゴリ特徴量の場合は頻出値を用いた単純な代入は、次元削減技術がデータセットのバリエーションをキーにしており、これらの単純な代入はそれを根本的に変えてしまうため、うまくいかないのです。(これについては、[この素晴らしいビデオ](https://www.youtube.com/watch?v=OOM8_FH6_8o&feature=youtu.be)をご覧ください。このビデオはPCAに焦点を当てていますが、提供される情報はこれらの技術全てに適用可能です)。
# MAGIC 
# MAGIC データを正しくインプットするためには、既存のデータの分布を調べ、特徴間の関係を利用して、投影を変更しない方法で、その分布から適切な値をインプットする必要があります。この分野の研究はまだ始まったばかりですが、PCAやMCAだけでなく、FAMDの仕組みを解明した統計学者がいます。 私たちの課題は、Pythonにこれらの技術を実装したライブラリがないことです。しかし、Rにはこのためのパッケージがあります。
# MAGIC 
# MAGIC そこで、SparkのSQLエンジンでデータを一時ビューとして取得することにしました。 これにより、Rからこのデータを照会できるようになります。

# COMMAND ----------

# DBTITLE 1,変換されたデータをSpark DataFrameとして登録する
# assemble full dataset with transformed features
trans_features_pd = pd.concat([ 
  households_pd,  # add household IDs as supplemental variable
  qc_features_pd, 
  features_pd[b_feature_names].astype(str)
  ], axis=1)

# send dataset to spark as temp table
spark.createDataFrame(trans_features_pd).createOrReplaceTempView('trans_features_pd')

# COMMAND ----------

# MAGIC %md Rの環境を整えるために、必要なパッケージをロードする。[FactoMineR](https://www.rdocumentation.org/packages/FactoMineR/versions/2.4) パッケージは必要なFAMD機能を提供し、[missMDA](https://www.rdocumentation.org/packages/missMDA/versions/1.18) パッケージはインピュテーション機能を提供する。

# COMMAND ----------

# DBTITLE 1,Install Required R Packages
# MAGIC %r
# MAGIC 
# MAGIC require(devtools)
# MAGIC # install_version("dplyr", version="1.0.3")
# MAGIC install.packages( c( "pbkrtest", "FactoMineR", "missMDA", "factoextra") )

# COMMAND ----------

# MAGIC %md データをSparkR DataFrameに取得してから、Rのローカルデータフレームに収集していることに注意してください。

# COMMAND ----------

# DBTITLE 1,SparkのデータをRのデータフレームに取得する
# MAGIC %r
# MAGIC 
# MAGIC # retrieve data from from Spark
# MAGIC library(SparkR)
# MAGIC df.spark <- SparkR::sql("SELECT * FROM trans_features_pd")
# MAGIC 
# MAGIC # move data to R data frame
# MAGIC df.r <- SparkR::collect(df.spark)
# MAGIC 
# MAGIC summary(df.r)

# COMMAND ----------

# MAGIC %md データはうまく伝わっているように見えますが、バイナリ特徴がどのように変換されたかを調べる必要があります。 FactoMinerとmissMDAはカテゴリ特徴を[*factor* types](https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor)として識別することを要求していますが、ここでは文字として伝わっていることが確認できます。

# COMMAND ----------

# DBTITLE 1,Rデータフレームの構造を調べる
# MAGIC %r
# MAGIC 
# MAGIC str(df.r)

# COMMAND ----------

# MAGIC %md カテゴリカルな特徴を因子に変換するために、簡単な変換を適用する。

# COMMAND ----------

# DBTITLE 1,カテゴリフィーチャーをファクターに変換する
# MAGIC %r
# MAGIC 
# MAGIC library(dplyr)
# MAGIC df.mutated <- mutate_if(df.r, is.character, as.factor)
# MAGIC 
# MAGIC str(df.mutated)

# COMMAND ----------

# MAGIC %md これでデータは分析に適した構造になったので、FAMDを実行する作業を始めることができる。最初のステップは、必要な主成分の数を決定することです。missMDAパッケージはこの目的のために *estim_ncpFAMD* メソッドを提供していますが、このルーチンは **完了するのに長い時間がかかる** ことに注意してください。 このルーチンを実行するために使用したコードを含みますが、コメントアウトして、最終的に実行中に得られた結果に置き換えています。

# COMMAND ----------

# DBTITLE 1,Component数の決定
# MAGIC %r
# MAGIC 
# MAGIC library(missMDA)
# MAGIC 
# MAGIC # determine number of components to produce
# MAGIC #nb <- estim_ncpFAMD(df.mutated, ncp.max=10, sup.var=1)
# MAGIC nb <- list( c(8) ) 
# MAGIC names(nb) <- c("ncp")
# MAGIC 
# MAGIC # display optimal number of components
# MAGIC nb$ncp

# COMMAND ----------

# MAGIC %md 主成分の数が決まれば、あとは欠損値をインピュートするだけです。 FAMDは、PCAやMCAと同様に、特徴量を標準化する必要があることに注意してください。 このメカニズムは、特徴が連続的かカテゴリ的かによって異なります。 *imputeFAMD*メソッドは、*scale* 引数の適切な設定により、これに取り組むための機能を提供します。

# COMMAND ----------

# DBTITLE 1,欠測値のインプットとFAMD変換の実行
# MAGIC %r 
# MAGIC 
# MAGIC # impute missing values
# MAGIC library(missMDA)
# MAGIC 
# MAGIC res.impute <- imputeFAMD(
# MAGIC   df.mutated,      # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,      # number of principal components
# MAGIC   scale=True,      # standardize features
# MAGIC   max.iter=10000,  # iterations to find optimal solution
# MAGIC   sup.var=1        # ignore the household_id field (column 1)
# MAGIC   ) 
# MAGIC 
# MAGIC # perform FAMD
# MAGIC library(FactoMineR)
# MAGIC 
# MAGIC res.famd <- FAMD(
# MAGIC   df.mutated,     # dataset with categoricals organized as factors
# MAGIC   ncp=nb$ncp,     # number of principal components
# MAGIC   tab.disj=res.impute$tab.disj, # imputation matrix from prior step
# MAGIC   sup.var=1       # ignore the household_id field (column 1)
# MAGIC   )

# COMMAND ----------

# MAGIC %md FAMDによって生成された各主成分は、データセット全体に見られる分散のパーセンテージを占めている。 各主成分のパーセントは、ディメンション1～8として識別され、主成分が占める累積分散とともにFAMDの出力に取り込まれる。

# COMMAND ----------

# DBTITLE 1,コンポーネントが捉えた分散をプロットする
# MAGIC %r
# MAGIC 
# MAGIC library("ggplot2")
# MAGIC library("factoextra")
# MAGIC 
# MAGIC eig.val <- get_eigenvalue(res.famd)
# MAGIC print(eig.val)

# COMMAND ----------

# MAGIC %md この出力を見てみると、最初の2次元（主成分）が分散の約50％を占めていることがわかり、2次元の可視化によってデータの構造を把握することができるようになりました。

# COMMAND ----------

# DBTITLE 1,2つの要素による世帯の可視化
# MAGIC %r
# MAGIC 
# MAGIC fviz_famd_ind(
# MAGIC   res.famd, 
# MAGIC   axes=c(1,2),  # use principal components 1 & 2
# MAGIC   geom = "point",  # show just the points (households)
# MAGIC   col.ind = "cos2", # color points (roughly) by the degree to which the principal component predicts the instance
# MAGIC   gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
# MAGIC   alpha.ind=0.5
# MAGIC   )

# COMMAND ----------

# MAGIC %md 第一主成分と第二主成分で世帯をグラフ化すると、データの中にいくつかの素晴らしい世帯のクラスターがあることがわかります（グラフのグループ化パターンが示すとおりです）。また、より低いレベルでは、より細かいクラスターが存在し、その境界が大きなグループと重なっている可能性があります。
# MAGIC 
# MAGIC FAMDの結果に対して、基本的な特徴がそれぞれの主成分でどのように表現されているかをより理解するために、[他にも多くの種類の可視化や分析が可能です](http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/115-famd-factor-analysis-of-mixed-data-in-r-essentials/)が、クラスタリングの目的には必要なものが揃っています。これから、RからPythonにデータを取り込むことに集中します。
# MAGIC 
# MAGIC まず、各世帯の主成分値を取得します。

# COMMAND ----------

# DBTITLE 1,主成分（固有値）に対する世帯固有の値を取得する
# MAGIC %r
# MAGIC 
# MAGIC df.famd <- bind_cols(
# MAGIC   select(df.r, "household_id"), 
# MAGIC   as.data.frame( res.famd$ind$coord ) 
# MAGIC   )
# MAGIC 
# MAGIC head(df.famd)

# COMMAND ----------

# DBTITLE 1,固有値の永続化(Deltaテーブルで保存)
# MAGIC %r
# MAGIC 
# MAGIC df.out <- createDataFrame(df.famd)
# MAGIC 
# MAGIC write.df(df.out, source = "delta", path = "/tmp/completejourney/silver/features_finalized", mode="overwrite", overwriteSchema="true")

# COMMAND ----------

# DBTITLE 1,保存した固有値が読み込めるか確認 (Python)
display(
  spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`')
  )

# COMMAND ----------

# MAGIC %md そして、次にこれらの特徴量の関係性を見てみましょう。

# COMMAND ----------

# DBTITLE 1,縮小された次元の関係を調べる
# generate correlations between features
famd_features_corr = spark.table('DELTA.`/tmp/completejourney/silver/features_finalized/`').drop('household_id').toPandas().corr()

# assemble a mask to remove top-half of heatmap
top_mask = np.zeros(famd_features_corr.shape, dtype=bool)
top_mask[np.triu_indices(len(top_mask))] = True

# define size of heatmap (for large number of features)
plt.figure(figsize=(10,8))

# generate heatmap
hmap = sns.heatmap(
  famd_features_corr,
  cmap = 'coolwarm',
  vmin =  1.0, 
  vmax = -1.0,
  mask = top_mask
  )

# COMMAND ----------

# MAGIC %md 特徴量の削減により多重共線性が解消されたので、クラスタリングを行うことができるようになりました。
