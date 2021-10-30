# Databricks notebook source
# MAGIC %md このノートブックの目的は、ワイドでディープな協調フィルタレコメンダーを構築するために使用する機能をエンジニアリングすることです。 このノートブックは **Databricks 8.1+ ML cluster** で実行する必要があります。

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
from pyspark.sql import functions as f

# COMMAND ----------

# MAGIC %md ## Step 1: 基準メトリックの計算
# MAGIC 
# MAGIC モデルベースの協調フィルタは、ユーザーと製品の特徴を利用して、将来の購入や交流を予測します。 [wide-and-deep model](https://arxiv.org/abs/1606.07792)は、顧客の将来の購入は、以前のユーザーと製品の相互作用や、ユーザーと製品の好みを取り巻く一般的なパターンの結果である可能性が高いと認識しています。これにより、特定の製品に対する特定のユーザーの好みと、新しい、つまり以前に購入されていない製品の購入に影響を与えるような、より一般的な好みとのバランスを取っています。 
# MAGIC 
# MAGIC モデルのワイドパートでは、ユーザーと商品のIDを利用して嗜好を記憶するという簡単な機能を持っています。 ディープパートのモデルでは、一般化を可能にするために、ユーザーと商品を説明する様々な特徴が必要です。 これらの特徴は、過去のデータから抽出されたメトリクスから導き出されますが、ここでは「*prior*評価セット」とラベル付けされています。

# COMMAND ----------

# DBTITLE 1,過去の注文を検索する
order_details_ = spark.table('instacart.order_details').cache()
prior_order_details = order_details_.filter(f.expr("eval_set='prior'"))

# COMMAND ----------

# MAGIC %md 多くの深層機能は、最後の注文から一定の日数前の注文に基づいて計算されます。 この間隔を30日前、180日前、360日前と任意に設定しています。

# COMMAND ----------

# DBTITLE 1,Days-Prior Boundariesの設定
prior_days = [30, 180, 360]

# COMMAND ----------

# MAGIC %md We can now calculate counts for various distinct elements observed within these prior windows.  These global metrics will be used to convert totals derived below into ratios in later steps. Because of the redundant nature of the metric definitions, we will iteratively construct these metrics before asking Spark to resolve them for us:

# COMMAND ----------

# DBTITLE 1,グローバルメトリクスの算出
# calculate metrics for the following fields and time intervals
aggregations = []
for column in ['order_id', 'user_id', 'product_id', 'department_id', 'aisle_id']:
  for prior_day in prior_days:
    
    # count distinct instances in the field during this time-range
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('global_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]
    
# execute metric definitions
global_metrics = (
  prior_order_details
  ).agg(*aggregations)

# show results
display(global_metrics)

# COMMAND ----------

# MAGIC %md それでは、プロダクト固有のメトリックの算出をしていきます。

# COMMAND ----------

# DBTITLE 1,プロダクト固有のメトリックの算出
# calculate metrics for the following fields and time intervals
aggregations = []

# distinct count metrics
for column in ['order_id', 'user_id']:
  for prior_day in prior_days:
    
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('product_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]

# occurrence count metrics
for column in ['reordered', 1]:
  for prior_day in prior_days:
    
    aggregations += [
      f.sum(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('product_sum_{1}_last_{0}_days'.format(prior_day, column))]
    
# get last assigned department & aisle for each product
  product_cat = (
    prior_order_details
      .select('product_id','aisle_id','department_id','order_id')
      .withColumn('aisle_id', f.expr('LAST(aisle_id) OVER(PARTITION BY product_id ORDER BY order_id)'))
      .withColumn('department_id', f.expr('LAST(department_id) OVER(PARTITION BY product_id ORDER BY order_id)'))
      .select('product_id','aisle_id','department_id')
      .distinct()
    )

# execute metric definitions
product_metrics = (
  prior_order_details
    .groupBy('product_id')
      .agg(*aggregations)
    .join(product_cat, on='product_id')
  )

# show results
display(product_metrics)

# COMMAND ----------

# MAGIC %md ユーザー固有のメトリックの算出をしていきます。

# COMMAND ----------

# DBTITLE 1,ユーザー固有のメトリックの算出
# calculate metrics for the following fields and time intervals
aggregations = []

# distinct count metrics
for column in ['order_id', 'product_id', 'department_id', 'aisle_id']:
  for prior_day in prior_days:
    
    aggregations += [
      f.countDistinct(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('user_cnt_distinct_{1}_last_{0}_days'.format(prior_day, column))]    

# occurrence count metrics
for column in ['reordered', 1]:
  for prior_day in prior_days:
    
    aggregations += [
      f.sum(
        f.expr(
          'CASE WHEN (days_prior_to_last_order <= {0}) THEN {1} ELSE NULL END'.format(prior_day, column))
        ).alias('user_sum_{1}_last_{0}_days'.format(prior_day, column))]
    
# execute metric definitions  
user_metrics = (
  prior_order_details
    .groupBy('user_id')
      .agg(*aggregations)
  )

# show results
display(user_metrics)

# COMMAND ----------

# MAGIC %md ## Step 2: 特徴量の計算
# MAGIC 
# MAGIC メトリクスが算出されたので、これらを使って製品固有の特徴を生成することができます。 製品固有の特徴はユーザー特徴とは別に保存し、後でデータを簡単に組み立てられるようにしておきます。

# COMMAND ----------

# DBTITLE 1,プロダクト固有のメトリックの算出
# calculate product specific features
product_feature_definitions = []
for prior_day in prior_days:
  
  # distinct users associated with a product within some number of prior days
  product_feature_definitions += [f.expr('product_cnt_distinct_user_id_last_{0}_days/global_cnt_distinct_user_id_last_{0}_days as product_shr_distinct_users_last_{0}_days'.format(prior_day))]
  
  # distinct orders associated with a product within some number of prior days
  product_feature_definitions += [f.expr('product_cnt_distinct_order_id_last_{0}_days/global_cnt_distinct_order_id_last_{0}_days as product_shr_distinct_orders_last_{0}_days'.format(prior_day))]
  
  # product reorders within some number of prior days
  product_feature_definitions += [f.expr('product_sum_reordered_last_{0}_days/product_sum_1_last_{0}_days as product_shr_reordered_last_{0}_days'.format(prior_day))]
  
# execute features
product_features = (
  product_metrics
    .join(global_metrics) # cross join to a single row
    .select(
      'product_id',
      'aisle_id',
      'department_id',
      *product_feature_definitions
      )
  ).na.fill(0) # fill any missing values with 0s

# persist data
(
product_features
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema','true')
  .saveAsTable('instacart.product_features')
)

# show results
display(spark.table('instacart.product_features'))

# COMMAND ----------

# MAGIC %md 同様に、ユーザー固有の特徴量を計算し、後で使用するためにこれらを永続させることもできます。

# COMMAND ----------

# DBTITLE 1,ユーザー固有のメトリックの算出
# calculate user-specific order metrics
median_cols = ['lines_per_order', 'days_since_prior_order']
approx_median_stmt = [f.expr(f'percentile_approx({col}, 0.5)').alias(f'user_med_{col}') for col in median_cols]

user_order_features = (
  prior_order_details
    .groupBy('user_id','order_id')  # get order-specific details for each user
      .agg(
        f.first('days_since_prior_order').alias('days_since_prior_order'),
        f.count('*').alias('lines_per_order')        
        )
    .groupBy('user_id') # get median values across user orders
      .agg(*approx_median_stmt)
  ).na.fill(0)

# calculate user overall features
user_feature_definitions = []
user_drop_columns = []

for prior_day in prior_days:
  user_feature_definitions += [f.expr('user_sum_reordered_last_{0}_days/user_sum_1_last_{0}_days as user_shr_reordered_last_{0}_days'.format(prior_day))]
  user_drop_columns += ['user_sum_reordered_last_{0}_days'.format(prior_day)]
  user_drop_columns += ['user_sum_1_last_{0}_days'.format(prior_day)]
  
# assemble final set of user features
user_features = (
  user_metrics
    .join(user_order_features, on=['user_id'])
    .select(
      f.expr('*'),
      *user_feature_definitions
      )
    .drop(*user_drop_columns)
  ).na.fill(0)

# persist data
(
user_features
  .write
  .format('delta')
  .mode('overwrite')
  .option('overwriteSchema','true')
  .saveAsTable('instacart.user_features')
)

# show user features
display(spark.table('instacart.user_features'))

# COMMAND ----------

# MAGIC %md # Step 3: ラベルを生成する
# MAGIC 
# MAGIC 次に、データセットで観測されたユーザーと製品のペアにラベルを付ける必要があります。 ここでは、ユーザーと商品の組み合わせのうち、そのレコードが顧客の最後の購入時、つまり「トレーニング期間」に購入されたものであれば「1」を、そうでなければ「0」を付けます。
# MAGIC 
# MAGIC **注** 私たちは、すべてのユーザーと製品の組み合わせを調べるのではなく、データセットを、前回またはトレーニング期間に発生した組み合わせに限定しました。 これは、他の人が自分のデータセットで再検討したいと思う選択です。

# COMMAND ----------

# DBTITLE 1,前回の購入時にユーザーと製品の組み合わせを特定する
train_labels = (
  order_details_
    .filter(f.expr("eval_set='train'"))
    .select('user_id', 'product_id')
    .distinct()
    .withColumn('label', f.lit(1))
     )

labels = (
  prior_order_details
    .select('user_id','product_id')
    .distinct()
    .join(train_labels, on=['user_id','product_id'], how='fullouter') # preserve all user-product combinations observed in either period
    .withColumn('label',f.expr('coalesce(label,0)'))
    .select('user_id','product_id','label')
    .withColumn('id', f.monotonically_increasing_id())
  )
  
(
  labels
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('instacart.labels')
  )
  
display(spark.table('instacart.labels'))

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
