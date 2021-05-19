# Databricks notebook source
# MAGIC %md 
# MAGIC このノートブックの目的は、協調フィルタリングによるレコメンダーを実施するために使用するデータセットを準備することです。 このノートブックは **Databricks 7.1+ クラスタ** で実行する必要があります。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC # イントロダクション
# MAGIC 
# MAGIC コラボレーティブ・フィルターは、現代のレコメンデーション体験を実現する重要な要素です。「 ***あなたに似たお客様はこんなものも買っています***」というタイプのレコメンデーションは、関連性の高いお客様の購買パターンに基づいて、興味を引きそうな商品を特定する重要な手段となります。
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/instacart_collabrecom.png" width="600">

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインポート
from pyspark.sql.types import *
from pyspark.sql.functions import count, countDistinct, avg, log, lit, expr

import shutil

# COMMAND ----------

# MAGIC %md # Step 1: データを読み込む
# MAGIC 
# MAGIC この種のレコメンデーションの基本的な構成要素は、顧客の取引データです。この種のデータを提供するために、私たちは人気のある[Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis)を使用します。このデータセットは、200,000人以上のInstacartユーザーが約50,000の商品の中から300万件以上の食料品を注文した際のカートレベルの詳細を提供しています。
# MAGIC 
# MAGIC **注意** データの提供条件により、この作品を再現するには、Kaggleからデータファイルをダウンロードし、以下のようなフォルダ構造にアップロードする必要があります。
# MAGIC 
# MAGIC 
# MAGIC ダウンロードしたデータファイルは以下のようにストレージ上(`/mnt/instacart`配下)に配置されているとします。
# MAGIC 
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これらのファイルをデータフレームに読み込み、次のようなデータモデルを構築し、お客様が個々の取引で購入した商品を把握して行きましょう。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC 
# MAGIC このデータには最小限の変換を実施した後、より高速なアクセスのためにDelta Lake形式で保存します。

# COMMAND ----------

# DBTITLE 1,サンプルデータをストレージ上に配置する(必要に応じて実行してください。)
# MAGIC %sh
# MAGIC 
# MAGIC #### サンプルデータをストレージ上に配置する
# MAGIC #### 必要に応じてコメントアウトを外して実行してください。
# MAGIC 
# MAGIC rm -rf /dbfs/tmp/mnt/instacart
# MAGIC rm -rf tmpdir
# MAGIC 
# MAGIC mkdir -p /dbfs/tmp/mnt/instacart/bronze/aisles
# MAGIC mkdir -p /dbfs/tmp/mnt/instacart/bronze/departments
# MAGIC mkdir -p /dbfs/tmp/mnt/instacart/bronze/order_products
# MAGIC mkdir -p /dbfs/tmp/mnt/instacart/bronze/orders
# MAGIC mkdir -p /dbfs/tmp/mnt/instacart/bronze/products
# MAGIC 
# MAGIC wget -nc 'https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/instacart-market-basket-analysis.zip'
# MAGIC unzip instacart-market-basket-analysis.zip -d tmpdir
# MAGIC 
# MAGIC cd tmpdir
# MAGIC 
# MAGIC unzip aisles.csv.zip
# MAGIC unzip departments.csv.zip
# MAGIC unzip order_products__prior.csv.zip 
# MAGIC unzip order_products__train.csv.zip
# MAGIC unzip orders.csv.zip 
# MAGIC unzip products.csv.zip
# MAGIC 
# MAGIC cp aisles.csv /dbfs/tmp/mnt/instacart/bronze/aisles/
# MAGIC cp departments.csv /dbfs/tmp/mnt/instacart/bronze/departments/
# MAGIC cp order_products__prior.csv /dbfs/tmp/mnt/instacart/bronze/order_products/
# MAGIC cp order_products__train.csv /dbfs/tmp/mnt/instacart/bronze/order_products/
# MAGIC cp orders.csv /dbfs/tmp/mnt/instacart/bronze/orders/
# MAGIC cp products.csv /dbfs/tmp/mnt/instacart/bronze/products/

# COMMAND ----------

# DBTITLE 1,databaseを作成する
_ = spark.sql('CREATE DATABASE IF NOT EXISTS instacart')

# COMMAND ----------

# MAGIC %md #重要
# MAGIC 
# MAGIC **注意** 受注データセットは、あらかじめ*prior*と*training*のデータセットに分割されています。 このデータセットの日付情報は非常に限られているので、あらかじめ定義された分割を使って作業する必要があります。 ここでは、*prior*データセットを***calibration***データセットとし、*training*データセットを***evaluation***データセットとします。混乱を避けるため、データ準備の段階でデータの名前を変更しておきます。

# COMMAND ----------

# DBTITLE 1,Orders (注文テーブル)
# 古いテーブルがあれば削除する
_ = spark.sql('DROP TABLE IF EXISTS instacart.orders')

# 同様に、古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/silver/orders', ignore_errors=True)

# 今回扱うデータのスキーマを定義(事前に分かっているものとする)
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

# CSVファイルからデータを読み込む
orders = (
  spark
    .read
    .csv(
      '/tmp/mnt/instacart/bronze/orders',
      header=True,
      schema=orders_schema
      )
  )

# "eval_set" エントリーの名前を変更
orders_transformed = (
  orders
    .withColumn('split', expr("CASE eval_set WHEN 'prior' THEN 'calibration' WHEN 'train' THEN 'evaluation' ELSE NULL END"))
    .drop('eval_set')
  )

# Deltaに書き出す(Deltaフォーマットとしてストレージに書き込む)
(
  orders_transformed
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/mnt/instacart/silver/orders')
  )

# SQLでもデータが参照できるようにテーブルに登録する(DeltaファイルとHiveメタストアの関連づけ)
_ = spark.sql('''
  CREATE TABLE instacart.orders
  USING DELTA
  LOCATION '/tmp/mnt/instacart/silver/orders'
  ''')

# 準備したデータを確認する
display(
  spark.table('instacart.orders')
)

# COMMAND ----------

# DBTITLE 1,Products (製品テーブル)
# 古いテーブルがあれば削除する
_ = spark.sql('DROP TABLE IF EXISTS instacart.products')

# 同様に、古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/silver/products', ignore_errors=True)

# 今回扱うデータのスキーマを定義(事前に分かっているものとする)
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

# CSVファイルからデータを読み込む
products = (
  spark
    .read
    .csv(
      '/tmp/mnt/instacart/bronze/products',
      header=True,
      schema=products_schema
      )
  )

# Deltaに書き出す(Deltaフォーマットとしてストレージに書き込む)
(
  products
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/mnt/instacart/silver/products')
  )

# SQLでもデータが参照できるようにテーブルに登録する(DeltaファイルとHiveメタストアの関連づけ)
_ = spark.sql('''
  CREATE TABLE instacart.products
  USING DELTA
  LOCATION '/tmp/mnt/instacart/silver/products'
  ''')

# 準備したデータを確認する
display(
  spark.table('instacart.products')
  )

# COMMAND ----------

# DBTITLE 1,Order Products (製品注文テーブル)
# 古いテーブルがあれば削除する
_ = spark.sql('DROP TABLE IF EXISTS instacart.order_products')

# 同様に、古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/silver/order_products', ignore_errors=True)

# 今回扱うデータのスキーマを定義(事前に分かっているものとする)
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

# CSVファイルからデータを読み込む
order_products = (
  spark
    .read
    .csv(
      '/tmp/mnt/instacart/bronze/order_products',
      header=True,
      schema=order_products_schema
      )
  )

# Deltaに書き出す(Deltaフォーマットとしてストレージに書き込む)
(
  order_products
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/mnt/instacart/silver/order_products')
  )

# SQLでもデータが参照できるようにテーブルに登録する(DeltaファイルとHiveメタストアの関連づけ)
_ = spark.sql('''
  CREATE TABLE instacart.order_products
  USING DELTA
  LOCATION '/tmp/mnt/instacart/silver/order_products'
  ''')

# 準備したデータを確認する
display(
  spark.table('instacart.order_products')
  )

# COMMAND ----------

# DBTITLE 1,Departments (部署テーブル)
# 古いテーブルがあれば削除する
_ = spark.sql('DROP TABLE IF EXISTS instacart.departments')

# 同様に、古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/silver/departments', ignore_errors=True)

# 今回扱うデータのスキーマを定義(事前に分かっているものとする)
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

# CSVファイルからデータを読み込む
departments = (
  spark
    .read
    .csv(
      '/tmp/mnt/instacart/bronze/departments',
      header=True,
      schema=departments_schema
      )
  )

# Deltaに書き出す(Deltaフォーマットとしてストレージに書き込む)
(
  departments
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/mnt/instacart/silver/departments')
  )

# SQLでもデータが参照できるようにテーブルに登録する(DeltaファイルとHiveメタストアの関連づけ)
_ = spark.sql('''
  CREATE TABLE instacart.departments
  USING DELTA
  LOCATION '/tmp/mnt/instacart/silver/departments'
  ''')

# 準備したデータを確認する
display(
  spark.table('instacart.departments')
  )

# COMMAND ----------

# DBTITLE 1,Aisles (通路テーブル)
# 古いテーブルがあれば削除する
_ = spark.sql('DROP TABLE IF EXISTS instacart.aisles')

# 同様に、古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/silver/aisles', ignore_errors=True)

# 今回扱うデータのスキーマを定義(事前に分かっているものとする)
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

# CSVファイルからデータを読み込む
aisles = (
  spark
  .read
  .csv(
    '/tmp/mnt/instacart/bronze/aisles',
    header=True,
    schema=aisles_schema
  )
)

# Deltaに書き出す(Deltaフォーマットとしてストレージに書き込む)
(
  aisles
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/mnt/instacart/silver/aisles')
  )

# SQLでもデータが参照できるようにテーブルに登録する(DeltaファイルとHiveメタストアの関連づけ)
_ = spark.sql('''
  CREATE TABLE instacart.aisles
  USING DELTA
  LOCATION '/tmp/mnt/instacart/silver/aisles'
  ''')

# 準備したデータを確認する
display(
  spark.table('instacart.aisles')
  )

# COMMAND ----------

# MAGIC %md # Step 2: 製品の *評価(Ratings)* を導き出す
# MAGIC 
# MAGIC 協調フィルタ（Collaborative Filter, CF）では、個々の製品に対するユーザーの好みを把握するための情報が必要になります。シナリオによっては、5つ星のうち3つ星のような明示的なユーザー評価が提供されることもあります。しかし、すべてのインタラクションに評価が付くわけではありません。また、多くのトランザクショナル・エンゲージメントでは、顧客にそのような評価を求めることができない場合が一般的です。このようなシナリオでは、製品の好みを示すために、他のユーザー生成データを使用することがあります。Instacartのデータセットの文脈では、ユーザーによる製品購入の頻度がそのような指標となるかもしれません。

# COMMAND ----------

# 古いDeltaファイルがあれば削除する
shutil.rmtree('/dbfs/tmp/mnt/instacart/gold/ratings__user_product_orders', ignore_errors=True)

# ユーザーごとに購入した回数を算出する
user_product_orders = (
  spark
  .table('instacart.orders') # テーブルデータを読み込んでDataFrameとして返す
  .join(spark.table('instacart.order_products'), on='order_id')
  .groupBy('user_id', 'product_id', 'split')
  .agg( count(lit(1)).alias('purchases') )
)

# Deltaで保存する(ストレージに書き出す)
(
  user_product_orders
  .write
  .format('delta')
  .mode('overwrite')
  .save('/tmp/mnt/instacart/gold/ratings__user_product_orders')
)

# 書き込んだ結果を結果をテーブル参照する
display(
  spark.sql('''
    SELECT * 
    FROM DELTA.`/tmp/mnt/instacart/gold/ratings__user_product_orders` 
    ORDER BY split, user_id, product_id
    ''')
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 製品購入を「推定評価値(暗黙の評価値)」として使用する場合、スケーリングの問題があります。 あるユーザーがある製品を10回購入し、別のユーザーが20回購入したとします。最初のユーザーは、その製品に対してより強い嗜好を持っているでしょうか？ もし、最初のユーザーが合計10回購入し、各チェックアウトイベントにその商品が含まれていたのに対し、2番目のユーザーは合計50回購入し、そのうち20回しか対象商品が含まれていなかったとしたらどうでしょうか？ この追加情報を踏まえて、ユーザーの好みについての理解・確度は変化するのでしょうか?
# MAGIC 
# MAGIC 
# MAGIC 購入頻度の違いを考慮してデータを再スケーリングすることで、より信頼性の高いユーザーの比較が可能になります。この方法にはいくつかのオプションがありますが、ユーザー間の類似性を測定する方法（協調フィルタリングの基礎となる）を考慮して、L2正規化と呼ばれる方法を使用してみましょう。
# MAGIC 
# MAGIC L2正規化を理解するために、製品XとYを購入した2人のユーザーを考えてみましょう。1人目のユーザーは、製品Xを10回、製品Yを5回購入しています。2番目のユーザーは、製品XとYをそれぞれ20回購入しています。 これらの購入を（x軸に製品X、y軸に製品Yを置いて）次のようにプロットすることができます。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm01.png' width=380>
# MAGIC 
# MAGIC 似ているかどうかを判断するには、この2つの軸の交点、つまり図中の2つの三角形の頂点の間の（ユークリッド）距離を測定します。 リスケールしない場合、1人目のユーザーは原点から約11ユニット、2人目のユーザーは約28ユニットの位置にいることになります。 この空間における2人のユーザー間の距離を計算すると、製品の好みや購入頻度の違いを測ることができます。空間の原点からの各ユーザーの距離を再スケーリングすることで、購入頻度に関する違いがなくなり、製品の好みの違いに焦点を当てることができます。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/lsh_norm02.png' width=400>
# MAGIC 
# MAGIC 再スケーリングは、各ユーザーと原点との間のユークリッド距離を計算し（この計算には2次元に限定する必要はありません）、そのユーザーの各製品固有の値を、L2ノルムと呼ばれるこの距離で割ることで達成されます。 ここでは、L2ノルムを推定評価値(暗黙の評価値、implied ratings)に適用します。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS instacart.user_ratings;
# MAGIC 
# MAGIC CREATE VIEW instacart.user_ratings 
# MAGIC AS
# MAGIC   WITH ratings AS (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       user_id,
# MAGIC       product_id,
# MAGIC       SUM(purchases) as purchases
# MAGIC     FROM DELTA.`/tmp/mnt/instacart/gold/ratings__user_product_orders`
# MAGIC     GROUP BY split, user_id, product_id
# MAGIC     )
# MAGIC   SELECT
# MAGIC     a.split,
# MAGIC     a.user_id,
# MAGIC     a.product_id,
# MAGIC     a.purchases,
# MAGIC     a.purchases/b.l2_norm as normalized_purchases
# MAGIC   FROM ratings a
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       user_id,
# MAGIC       POW( 
# MAGIC         SUM(POW(purchases,2)),
# MAGIC         0.5
# MAGIC         ) as l2_norm
# MAGIC     FROM ratings
# MAGIC     GROUP BY user_id, split
# MAGIC     ) b
# MAGIC     ON a.user_id=b.user_id AND a.split=b.split;
# MAGIC   
# MAGIC SELECT * FROM instacart.user_ratings ORDER BY user_id, split, product_id;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC これらの計算を(結果を新しい`table`で作成するのではなく)`view`で実装した点について、疑問に思われたかもしれません。ユーザーの値は、そのユーザーの購入イベントごとに再計算する必要があります。なぜなら、そのイベントは、それぞれの「暗黙評価値」を調整するL2ノルムの値に影響を与えるからです。基本となる *ratings* テーブルに生の購入数を保持することで、ユーザーの全購入履歴を辿ることなく、このテーブルに新しい情報を段階的に追加することができます。 そのテーブルの値をビューで集約して正規化することで、少ないETL作業で正規化されたデータを簡単に抽出することができます。
# MAGIC 
# MAGIC 
# MAGIC これらの計算にどのデータを含めるかを検討することが重要です。お客様のシナリオによっては、これらの *推定評価値* の元となる取引履歴を、表明された好みがレコメンダーが使用される期間におけるユーザーの好みと一致する期間に限定することが適切な場合があります。 あるシナリオでは、これは履歴データを月、四半期、年などに限定することを意味します。 他のシナリオでは、これは、現在または差し迫った期間と同等の季節成分を持つ期間に過去のデータを制限することを意味する場合がある。 例えば、あるユーザーは、秋にはパンプキンスパイスフレーバーの製品を強く好むが、夏の間はあまり好まないかもしれません。 デモのために、ここでは取引履歴全体を評価の基礎として使用しますが、実際の実装ではこの点を慎重に検討する必要があります。

# COMMAND ----------

# MAGIC %md # Step 3: Derive Naive Product *Ratings*
# MAGIC 
# MAGIC A common practice when evaluating a recommender is to compare it to a prior or alternative recommendation engine to see which better helps the organization achieve its goals. To provide us a starting point for such comparisons, we might consider using overall product popularity as the basis for making *naive* collaborative recommendations. Here, we calculate normalized product ratings based on overall purchase frequencies to enable this work:
# MAGIC 
# MAGIC レコメンダーを評価する際の一般的な方法は、先行するレコメンダーや代替のレコメンダーと比較して、どちらが組織の目標達成に役立つかを確認することである。このような比較を行うための出発点として、製品全体の人気度を、*naive* 協調推薦の基底として使用することを検討しましょう。ここでは、この作業を可能にするために、全体的な購入頻度に基づいて正規化された製品評価を計算します。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS instacart.naive_ratings;
# MAGIC 
# MAGIC CREATE VIEW instacart.naive_ratings 
# MAGIC AS
# MAGIC   WITH ratings AS (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       product_id,
# MAGIC       SUM(purchases) as purchases
# MAGIC     FROM DELTA.`/tmp/mnt/instacart/gold/ratings__user_product_orders`
# MAGIC     GROUP BY split, product_id
# MAGIC     )
# MAGIC   SELECT
# MAGIC     a.split,
# MAGIC     a.product_id,
# MAGIC     a.purchases,
# MAGIC     a.purchases/b.l2_norm as normalized_purchases
# MAGIC   FROM ratings a
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       split,
# MAGIC       POW( 
# MAGIC         SUM(POW(purchases,2)),
# MAGIC         0.5
# MAGIC         ) as l2_norm
# MAGIC     FROM ratings
# MAGIC     GROUP BY split
# MAGIC     ) b
# MAGIC     ON a.split=b.split;
# MAGIC   
# MAGIC SELECT * FROM instacart.naive_ratings ORDER BY split, product_id;
