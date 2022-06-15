# Databricks notebook source
# MAGIC %md  このノートブックの目的は、セグメンテーション作業に必要なデータにアクセスし、準備することです。このノートブックはDatabricks ML 8.0 CPUベースのクラスタ上で開発されました。

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
from pyspark.sql.functions import min, max

# COMMAND ----------

# MAGIC %md ## ステップ1：データへのアクセス
# MAGIC 
# MAGIC この演習の目的は、プロモーションの反応性に基づいて顧客世帯をセグメント化することに興味を持つプロモーション管理チームが、分析の部分をどのように実行するかを示すことです。 今回使用するデータセットは、DunnhumbyがKaggleを通じて公開しているもので、[*The Complete Journey*](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey)と呼ばれているものです。これは、約2年間にわたる約2,500世帯の様々なプロモーションキャンペーンと組み合わせた世帯の購買活動を特定する多数のファイルで構成されています。データセット全体のスキーマは以下のように表現される。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png' width=500>.
# MAGIC 
# MAGIC このデータを分析に利用するために、`/mnt/completejourney` という名前の[クラウドストレージマウントポイント](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)の*bronze*フォルダにダウンロード、抽出、ロードしています。 そこから、以下のようにデータを準備することが考えられます。

# COMMAND ----------

# DBTITLE 1,データベース作成
# MAGIC %sql
# MAGIC 
# MAGIC DROP DATABASE IF EXISTS journey CASCADE;
# MAGIC CREATE DATABASE journey;

# COMMAND ----------

# DBTITLE 1,トランザクション
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.transactions')

# expected structure of the file
transactions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('basket_id', LongType()),
  StructField('day', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('quantity', IntegerType()),
  StructField('sales_amount', FloatType()),
  StructField('store_id', IntegerType()),
  StructField('discount_amount', FloatType()),
  StructField('transaction_time', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('coupon_discount', FloatType()),
  StructField('coupon_discount_match', FloatType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/transaction_data.csv',
      header=True,
      schema=transactions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/transactions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.transactions 
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/transactions'
    ''')

# show data
display(
  spark.table('journey.transactions')
  )

# COMMAND ----------

# DBTITLE 1,プロダクト
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.products')

# expected structure of the file
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('manufacturer', StringType()),
  StructField('department', StringType()),
  StructField('brand', StringType()),
  StructField('commodity', StringType()),
  StructField('subcommodity', StringType()),
  StructField('size', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/product.csv',
      header=True,
      schema=products_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/products')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.products
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/products'
    ''')

# show data
display(
  spark.table('journey.products')
  )

# COMMAND ----------

# DBTITLE 1,世帯
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.households')

# expected structure of the file
households_schema = StructType([
  StructField('age_bracket', StringType()),
  StructField('marital_status', StringType()),
  StructField('income_bracket', StringType()),
  StructField('homeownership', StringType()),
  StructField('composition', StringType()),
  StructField('size_category', StringType()),
  StructField('child_category', StringType()),
  StructField('household_id', IntegerType())
  ])

# read data to dataframe
households = (
  spark
    .read
    .csv(
      '/tmp/completejourney/bronze/hh_demographic.csv',
      header=True,
      schema=households_schema
      )
  )

# make queriable for later work
households.createOrReplaceTempView('households')

# income bracket sort order
income_bracket_lookup = (
  spark.createDataFrame(
    [(0,'Under 15K'),
     (15,'15-24K'),
     (25,'25-34K'),
     (35,'35-49K'),
     (50,'50-74K'),
     (75,'75-99K'),
     (100,'100-124K'),
     (125,'125-149K'),
     (150,'150-174K'),
     (175,'175-199K'),
     (200,'200-249K'),
     (250,'250K+') ],
    schema=StructType([
            StructField('income_bracket_numeric',IntegerType()),
            StructField('income_bracket', StringType())
            ])
    )
  )

# make queriable for later work
income_bracket_lookup.createOrReplaceTempView('income_bracket_lookup')

# household composition sort order
composition_lookup = (
  spark.createDataFrame(
    [ (0,'Single Female'),
      (1,'Single Male'),
      (2,'1 Adult Kids'),
      (3,'2 Adults Kids'),
      (4,'2 Adults No Kids'),
      (5,'Unknown') ],
    schema=StructType([
            StructField('sort_order',IntegerType()),
            StructField('composition', StringType())
            ])
    )
  )

# make queriable for later work
composition_lookup.createOrReplaceTempView('composition_lookup')

# persist data with sort order data and a priori segments
(
  spark
    .sql('''
      SELECT
        a.household_id,
        a.age_bracket,
        a.marital_status,
        a.income_bracket,
        COALESCE(b.income_bracket_numeric, -1) as income_bracket_alt,
        a.homeownership,
        a.composition,
        COALESCE(c.sort_order, -1) as composition_sort_order,
        a.size_category,
        a.child_category
      FROM households a
      LEFT OUTER JOIN income_bracket_lookup b
        ON a.income_bracket=b.income_bracket
      LEFT OUTER JOIN composition_lookup c
        ON a.composition=c.composition
      ''')
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/households')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.households 
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/households'
    ''')

# show data
display(
  spark.table('journey.households')
  )

# COMMAND ----------

# DBTITLE 1,クーポン
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.coupons')

# expected structure of the file
coupons_schema = StructType([
  StructField('coupon_upc', StringType()),
  StructField('product_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon.csv',
      header=True,
      schema=coupons_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/coupons')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.coupons
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/coupons'
    ''')

# show data
display(
  spark.table('journey.coupons')
  )

# COMMAND ----------

# DBTITLE 1,キャンペーン
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.campaigns')

# expected structure of the file
campaigns_schema = StructType([
  StructField('description', StringType()),
  StructField('campaign_id', IntegerType()),
  StructField('start_day', IntegerType()),
  StructField('end_day', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_desc.csv',
      header=True,
      schema=campaigns_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/campaigns')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.campaigns
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/campaigns'
    ''')

# show data
display(
  spark.table('journey.campaigns')
  )

# COMMAND ----------

# DBTITLE 1,クーポンの利用
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.coupon_redemptions')

# expected structure of the file
coupon_redemptions_schema = StructType([
  StructField('household_id', IntegerType()),
  StructField('day', IntegerType()),
  StructField('coupon_upc', StringType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/coupon_redempt.csv',
      header=True,
      schema=coupon_redemptions_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/coupon_redemptions')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.coupon_redemptions
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/coupon_redemptions'
    ''')

# show data
display(
  spark.table('journey.coupon_redemptions')
  )

# COMMAND ----------

# DBTITLE 1,キャンペーンと世帯の関係
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.campaigns_households')

# expected structure of the file
campaigns_households_schema = StructType([
  StructField('description', StringType()),
  StructField('household_id', IntegerType()),
  StructField('campaign_id', IntegerType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/campaign_table.csv',
      header=True,
      schema=campaigns_households_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/campaigns_households')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.campaigns_households
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/campaigns_households'
    ''')

# show data
display(
  spark.table('journey.campaigns_households')
  )

# COMMAND ----------

# DBTITLE 1,因果関係データ
# delete the old table if needed
_ = spark.sql('DROP TABLE IF EXISTS journey.causal_data')

# expected structure of the file
causal_data_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('store_id', IntegerType()),
  StructField('week_no', IntegerType()),
  StructField('display', StringType()),
  StructField('mailer', StringType())
  ])

# read data to dataframe
( spark
    .read
    .csv(
      '/tmp/completejourney/bronze/causal_data.csv',
      header=True,
      schema=causal_data_schema
      )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save('/tmp/completejourney/silver/causal_data')
  )

# create table object to make delta lake queriable
_ = spark.sql('''
    CREATE TABLE journey.causal_data
    USING DELTA 
    LOCATION '/tmp/completejourney/silver/causal_data'
    ''')

# show data
display(
  spark.table('journey.causal_data')
  )

# COMMAND ----------

# MAGIC %md ## ステップ2：トランザクションデータの調整
# MAGIC 
# MAGIC 生データがロードされたので、トランザクションデータを調整する必要があります。 このデータセットは小売業者が管理するキャンペーンに焦点を当てていますが、クーポン割引のマッチング情報を含めることで、小売業者とメーカーが作成したクーポンの両方から発生した割引をトランザクションデータに反映させることができます。 特定の商品トランザクションを特定のクーポンにリンクする機能がない場合（償還が行われた場合）、0 以外の *coupon_discount_match* 値に関連する *coupon_discount* 値は、メーカーのクーポンに由来するものと仮定します。 その他のクーポン割引はすべて、小売業者が作成したクーポンによるものと仮定する。 
# MAGIC 
# MAGIC 小売店クーポン割引とメーカークーポン割引の分離に加え、商品の定価は、売上金額から適用されるすべての割引を差し引いた金額として計算されます。

# COMMAND ----------

# DBTITLE 1,Adjusted Transactions
# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS journey.transactions_adj;
# MAGIC 
# MAGIC CREATE TABLE journey.transactions_adj
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     household_id,
# MAGIC     basket_id,
# MAGIC     week_no,
# MAGIC     day,
# MAGIC     transaction_time,
# MAGIC     store_id,
# MAGIC     product_id,
# MAGIC     amount_list,
# MAGIC     campaign_coupon_discount,
# MAGIC     manuf_coupon_discount,
# MAGIC     manuf_coupon_match_discount,
# MAGIC     total_coupon_discount,
# MAGIC     instore_discount,
# MAGIC     amount_paid,
# MAGIC     units
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       household_id,
# MAGIC       basket_id,
# MAGIC       week_no,
# MAGIC       day,
# MAGIC       transaction_time,
# MAGIC       store_id,
# MAGIC       product_id,
# MAGIC       COALESCE(sales_amount - discount_amount - coupon_discount - coupon_discount_match,0.0) as amount_list,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) = 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as campaign_coupon_discount,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_discount_match,0.0) != 0.0 THEN -1 * COALESCE(coupon_discount,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as manuf_coupon_discount,
# MAGIC       -1 * COALESCE(coupon_discount_match,0.0) as manuf_coupon_match_discount,
# MAGIC       -1 * COALESCE(coupon_discount - coupon_discount_match,0.0) as total_coupon_discount,
# MAGIC       COALESCE(-1 * discount_amount,0.0) as instore_discount,
# MAGIC       COALESCE(sales_amount,0.0) as amount_paid,
# MAGIC       quantity as units
# MAGIC     FROM journey.transactions
# MAGIC     );
# MAGIC     
# MAGIC SELECT * FROM journey.transactions_adj;

# COMMAND ----------

# MAGIC %md ## ステップ3: データを調べる
# MAGIC 
# MAGIC このデータセットのレコードの正確な開始日と終了日は不明である。 その代わり、日数は1〜711の値で表され、データセットの開始からの日数を示していると思われる。

# COMMAND ----------

# DBTITLE 1,Household Data in Transactions
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT household_id) as uniq_households_in_transactions,
# MAGIC   MIN(day) as first_day,
# MAGIC   MAX(day) as last_day
# MAGIC FROM journey.transactions_adj;

# COMMAND ----------

# MAGIC %md この分析の主な焦点は、ターゲットとなるメールマガジンやクーポンを含むと想定される様々な小売業者のキャンペーンに、世帯がどのように反応するかにあります。取引データセットのすべての世帯がキャンペーンのターゲットになっているわけではありませんが、ターゲットになっている世帯はすべて取引データセットに含まれています。

# COMMAND ----------

# DBTITLE 1,キャンペーンにおける世帯のデータ
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT a.household_id) as uniq_households_in_transactions,
# MAGIC   COUNT(DISTINCT b.household_id) as uniq_households_in_campaigns,
# MAGIC   COUNT(CASE WHEN a.household_id==b.household_id THEN 1 ELSE NULL END) as uniq_households_in_both
# MAGIC FROM (SELECT DISTINCT household_id FROM journey.transactions_adj) a
# MAGIC FULL OUTER JOIN (SELECT DISTINCT household_id FROM journey.campaigns_households) b
# MAGIC   ON a.household_id=b.household_id

# COMMAND ----------

# MAGIC %md キャンペーンの一環としてクーポンが世帯に送られると、データにはそのクーポンがどの商品と関連づけられたかが示されます。*coupon_redemptions* テーブルは、特定の世帯がどのクーポンをどの日に利用したかの詳細を提供します。しかし、クーポンそのものは、取引項目で識別されません。
# MAGIC 
# MAGIC このため、特定の品目とクーポンの関連付けを行い、取引を特定のキャンペーンに関連付けるのではなく、キャンペーンで販売された商品に関連するすべての品目を、クーポンの引き換えがあったかどうかにかかわらず、キャンペーンの影響を受けたものとして単純に属性付けすることにしました。 これは少しずさんですが、全体のロジックを単純化するためにこのようにしています。実際のデータ分析では、**これは再検討されるべき簡略化です**。さらに、我々は（*causal_data*テーブルで捕捉されるように）店頭ディスプレイや店舗固有のチラシの影響を調査していないことに注意してください。 これも、分析を単純化するためです。
# MAGIC 
# MAGIC ここに示すロジックは、キャンペーンと商品購入をどのように関連付けるかを示しており、当社の機能エンジニアリングノートブックで再現される予定です。

# COMMAND ----------

# DBTITLE 1,Transaction Line Items Flagged for Promotional Influences
# MAGIC %sql
# MAGIC 
# MAGIC WITH 
# MAGIC     targeted_products_by_household AS (
# MAGIC       SELECT DISTINCT
# MAGIC         b.household_id,
# MAGIC         c.product_id
# MAGIC       FROM journey.campaigns a
# MAGIC       INNER JOIN journey.campaigns_households b
# MAGIC         ON a.campaign_id=b.campaign_id
# MAGIC       INNER JOIN journey.coupons c
# MAGIC         ON a.campaign_id=c.campaign_id
# MAGIC       )
# MAGIC SELECT
# MAGIC   a.household_id,
# MAGIC   a.day,
# MAGIC   a.basket_id,
# MAGIC   a.product_id,
# MAGIC   CASE WHEN a.campaign_coupon_discount > 0 THEN 1 ELSE 0 END as campaign_coupon_redemption,
# MAGIC   CASE WHEN a.manuf_coupon_discount > 0 THEN 1 ELSE 0 END as manuf_coupon_redemption,
# MAGIC   CASE WHEN a.instore_discount > 0 THEN 1 ELSE 0 END as instore_discount_applied,
# MAGIC   CASE WHEN b.brand = 'Private' THEN 1 ELSE 0 END as private_label,
# MAGIC   CASE WHEN c.product_id IS NULL THEN 0 ELSE 1 END as campaign_targeted
# MAGIC FROM journey.transactions_adj a
# MAGIC INNER JOIN journey.products b
# MAGIC   ON a.product_id=b.product_id
# MAGIC LEFT OUTER JOIN targeted_products_by_household c
# MAGIC   ON a.household_id=c.household_id AND 
# MAGIC      a.product_id=c.product_id

# COMMAND ----------

# MAGIC %md 最後にもう一つ、このデータセットには、取引履歴で見つかった2,500世帯のうち、約800世帯の人口統計データが含まれているに過ぎないことに注意したい。これらのデータはプロファイリングを行う上で有用ですが、このような少ないサンプルデータから結論を導き出す前に注意が必要です。
# MAGIC 
# MAGIC 同様に、この2,500世帯がどのように抽出されたのか、その詳細も不明である。 今回の分析から得られたすべての結論は、この限界を認識した上で見る必要がある。

# COMMAND ----------

# DBTITLE 1,Households with Demographic Data
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   COUNT(DISTINCT a.household_id) as uniq_households_in_transactions,
# MAGIC   COUNT(DISTINCT b.household_id) as uniq_households_in_campaigns,
# MAGIC   COUNT(DISTINCT c.household_id) as uniq_households_in_households,
# MAGIC   COUNT(CASE WHEN a.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_transactions_households,
# MAGIC   COUNT(CASE WHEN b.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_campaigns_households,
# MAGIC   COUNT(CASE WHEN a.household_id==c.household_id AND b.household_id==c.household_id THEN 1 ELSE NULL END) as uniq_households_in_all
# MAGIC FROM (SELECT DISTINCT household_id FROM journey.transactions_adj) a
# MAGIC LEFT OUTER JOIN (SELECT DISTINCT household_id FROM journey.campaigns_households) b
# MAGIC   ON a.household_id=b.household_id
# MAGIC LEFT OUTER JOIN journey.households c
# MAGIC   ON a.household_id=c.household_id
