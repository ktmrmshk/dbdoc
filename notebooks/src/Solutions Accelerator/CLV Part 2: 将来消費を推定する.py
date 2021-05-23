# Databricks notebook source
# MAGIC %md ##将来消費を推定する
# MAGIC 
# MAGIC 前回のノートでは、定額制ではないモデルの顧客が、時間の経過とともにどのように離脱していくかを検証しました。 小売企業と顧客の間に書面契約がない場合、顧客が継続的な関係から脱落する確率は、他の顧客と比較した過去のエンゲージメントパターンに基づいて推定するしかありません。顧客が積極的に関与し続ける確率を理解することは、それ自体が非常に価値のあることです。 しかし、さらに一歩進んで、予測される将来のエンゲージメントからどれだけの収益や利益が得られるかを計算することができます。
# MAGIC 
# MAGIC そのためには、将来の購入イベントに関連する金銭的価値を計算するモデルを構築する必要があります。 このモデルの目的は、そのようなモデルを導き出し、生涯確率と組み合わせて推定顧客生涯価値を導き出すことです。

# COMMAND ----------

# MAGIC %md ###Step 1: 環境の設定
# MAGIC 
# MAGIC これまでと同様に、クラスタに以下のライブラリを[インストール](https://docs.databricks.com/libraries.html#workspace-libraries)・[アタッチ](https://docs.databricks.com/libraries.html#install-a-library-on-a-cluster)する必要があります。また、クラスタのランライムには **Databricks ML runtime** ver 6.5以上を指定する必要があります。:</p>
# MAGIC 
# MAGIC * xlrd
# MAGIC * lifetimes==0.10.1
# MAGIC * nbconvert
# MAGIC 
# MAGIC さらに、前のノートブックと同様に、UCI Machine Learning Repositoryから入手できる[Online Retail Data Set](http://archive.ics.uci.edu/ml/datasets/Online+Retail)を、 `/FileStore/tables/online_retail/`　フォルダにロードする必要があります。
# MAGIC 
# MAGIC (訳者注: このNotebookではすでにデータ形式が利用しやすいCSV形式になっているので、xlrd, nbconvertはインストール不要になっています。)

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリのインストール
# MAGIC %pip install lifetimes==0.10.1

# COMMAND ----------

# MAGIC %sh
# MAGIC wget 'https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/Online_Retail-93ac4.csv'
# MAGIC cp Online_Retail-93ac4.csv /dbfs/FileStore/tables/online_retail/

# COMMAND ----------

import pandas as pd
import numpy as np

# 対象のxlsxファイルのパスを取得
xlsx_filename = dbutils.fs.ls('file:///dbfs/FileStore/tables/online_retail')[0][0]

# 上記ファイルのデータのスキーマを設定(既知とする)
orders_schema = {
  'InvoiceNo':np.str,
  'StockCode':np.str,
  'Description':np.str,
  'Quantity':np.int64,
#  'InvoiceDate':np.datetime64,
  'InvoiceDate':np.str,
  'UnitPrice':np.float64,
  'CustomerID':np.str,
  'Country':np.str  
  }

#　元のファイルがCSVになっているので、そのまま読み込む
orders_pd = pd.read_csv(
  xlsx_filename, 
  sep=',',
  #sheet_name='Online Retail',
  header=0, # 第一行目はヘッダーになっている
  dtype=orders_schema,
  parse_dates=['InvoiceDate']
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 顧客の購入に関連する金銭的価値を調べるためには、オンライン小売注文データセットの各販売額を計算する必要があります。そのため、QuantityにUnitPriceを乗じて、新しいSalesAmountフィールドを作成します。

# COMMAND ----------

# 売上を算出: SalesAmount =  quantity * unit price
orders_pd['SalesAmount'] = orders_pd['Quantity'] * orders_pd['UnitPrice']

orders_pd.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC 上記のPandasデータをSpark上で使えるように準備する

# COMMAND ----------

# データ変換: pandas DF から Spark DF　へ
orders = spark.createDataFrame(orders_pd)

# SparkDFをクエリで使うために"orders"という名前のTemp Viewを作成
orders.createOrReplaceTempView('orders') 

# COMMAND ----------

# MAGIC %md ###Step 2: データの探索
# MAGIC 
# MAGIC データセットの購入頻度のパターンを調べるには、前のノートブックのステップ2のセクションを参照してください。 ここでは、顧客の消費に関するパターンを調べたいと思います。 
# MAGIC 
# MAGIC はじめに、顧客の典型的な1日あたりの購入額を見てみましょう。 顧客生涯の計算と同様に、同じ日に複数回の購入があっても同じ購入イベントとみなすため、1日単位でグループ化します。

# COMMAND ----------

# MAGIC %sql -- 顧客毎の日毎の購入額
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 1日の消費額の範囲は非常に広く、1日に£70,000以上を購入する顧客もいます。背景にあるビジネスの知識がないため、この段階で支出が期待値と一致しているのか、それとも排除すべき異常値なのかを判断することはできません。 
# MAGIC 
# MAGIC また、マイナスの値が非常に多いことにも注目してください。これはリターンに関連している可能性が高いです。 この点については後ほど詳しく説明しますが、今回はサイトで観察されるアクティビティの分布を把握するために、調べる範囲を狭めます。

# COMMAND ----------

# MAGIC %sql -- 顧客毎の日毎の購入額 (+条件: 日毎の売上高: 0 - 2500ポンド)
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)
# MAGIC HAVING SalesAmount BETWEEN 0 AND 2500

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC この限定した範囲での1日の支出額の分布は、200 -- 400ポンドを中心とし、それ以上の支出額に向かってロングテールになっています。これが正規分布（ガウス分布）でないことは明らかです。
# MAGIC 
# MAGIC このような分布パターンは、個々の顧客の支出パターンにも見られます。購入回数の多い顧客に焦点を当てると、支出パターンは様々ですが、この右に偏ったパターンが続いていることがわかります。

# COMMAND ----------

# MAGIC %sql -- 買い物頻度による顧客ランキング (Top5)
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   COUNT(DISTINCT TO_DATE(InvoiceDate)) as Frequency
# MAGIC FROM orders
# MAGIC GROUP BY CustomerID
# MAGIC ORDER BY Frequency DESC
# MAGIC LIMIT 5

# COMMAND ----------

# MAGIC %sql -- 上記の結果から、上位3顧客の日毎の購入額の分布を調べる
# MAGIC 
# MAGIC SELECT
# MAGIC   CustomerID,
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   SUM(SalesAmount) as SalesAmount
# MAGIC FROM orders
# MAGIC WHERE CustomerID IN (14911, 12748, 17841)
# MAGIC GROUP BY CustomerID, TO_DATE(InvoiceDate)
# MAGIC ORDER BY CustomerID

# COMMAND ----------

# MAGIC %md 
# MAGIC このデータセットには、もう少し検討すべき点があります。その中でも、まずは顧客ごとの指標を算出する必要があります。

# COMMAND ----------

# MAGIC %md ###Step 3: 顧客メトリックを算出する
# MAGIC 
# MAGIC 今回のデータセットには、生の取引履歴が含まれています。 前回と同様に、頻度(Frequency)、Age(T)、直近性(Recency)という顧客ごとの指標の計算が必要です。また同時に、金銭的価値の指標の算出も必要になります。:</p>
# MAGIC 
# MAGIC * **Frequency** - 観測期間中の取引(買い物)回数。ただし、初回購入は除く。つまり、(全取引回数 - 1)。日毎にカウント。つまり、同日に複数回取引があっても1回とカウントする。
# MAGIC * **Age (T)** - 経過日数, 初めての取引発生した日から現在の日付（またはデータセットの最終の日)
# MAGIC * **Recency** - 直近の取引があった時点のAge。つまり、初回の取引の日から直近(最後の)取引があった日までの経過日数。
# MAGIC * **Monetary Value** - 金銭的価値。顧客がリピート購入する際の1トランザクションあたりの平均消費額(マージンやその他の金銭的価値がある場合は、それを代用することも可能です。)
# MAGIC 
# MAGIC 顧客年齢などの指標を計算する際には、データセットがいつ終了するかを考慮する必要があることに注意してください。 これらの指標を今日の日付を基準にして計算すると、誤った結果になる可能性があります。 そこで、データセットの最後の日付を特定し、それをすべての計算において「*今日の日付*」と定義します。
# MAGIC 
# MAGIC これらのメトリクスを導き出すために、[lifetimesライブラリ](https://lifetimes.readthedocs.io/en/latest/lifetimes.html)の組み込み機能を利用することができます。前のノートブックのコードと同様、呼び出されているメソッドはほぼ同じものを使用しています。唯一の違いは、金額の尺度としてSalesAmountフィールドを使用するようにメソッドで指定していることです。

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリによる算出
import lifetimes

# 最後のトランザクション発生日をデータセットのエンドポイント(=「今日」)と見なす。
current_date = orders_pd['InvoiceDate'].max()

# 必要な顧客メトリックをlifetimesライブラリを使って算出する
metrics_pd = (
  lifetimes.utils.summary_data_from_transaction_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date, 
    freq='D',
    monetary_value_col='SalesAmount'  # use sales amount to determine monetary value
    )
  )

# 最初の数行を確認する
metrics_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 前回と同様に、大規模なデータセットを扱う際を想定して、これらの値を並列処理で計算できるようにSparkを使って同じデータセットを生成する方法を見ていきます。これには以下の2通りの方法があります。 
# MAGIC 
# MAGIC 1. SQLステートメントを使用する方法
# MAGIC 1. Python(プログラマティックSQL API)を使用する方法
# MAGIC 
# MAGIC これら2つを順に見ていきます。
# MAGIC 
# MAGIC コードは可能な限り前のノートブックと一貫性を保つようにしましたが、金額ロジックに必要な追加ロジックは一部そうでない部分があります。

# COMMAND ----------

# DBTITLE 1,SQLを用いた算出
# SQL文を記述
sql = '''
  SELECT
    a.customerid as CustomerID,
    CAST(COUNT(DISTINCT a.transaction_at) - 1 as float) as frequency,
    CAST(DATEDIFF(MAX(a.transaction_at), a.first_at) as float) as recency,
    CAST(DATEDIFF(a.current_dt, a.first_at) as float) as T,
    CASE                                              -- MONETARY VALUE CALCULATION
      WHEN COUNT(DISTINCT a.transaction_at)=1 THEN 0    -- 0 if only one order
      ELSE
        SUM(
          CASE WHEN a.first_at=a.transaction_at THEN 0  -- daily average of all but first order
          ELSE a.salesamount
          END
          ) / (COUNT(DISTINCT a.transaction_at)-1)
      END as monetary_value    
  FROM ( -- customer order history
    SELECT
      x.customerid,
      z.first_at,
      x.transaction_at,
      y.current_dt,
      x.salesamount                  
    FROM (                                            -- customer daily summary
      SELECT 
        customerid, 
        TO_DATE(invoicedate) as transaction_at, 
        SUM(SalesAmount) as salesamount               -- SALES AMOUNT ADDED
      FROM orders 
      GROUP BY customerid, TO_DATE(invoicedate)
      ) x
    CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
    INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
      ON x.customerid=z.customerid
    WHERE x.customerid IS NOT NULL
    ) a
  GROUP BY a.customerid, a.current_dt, a.first_at
  ORDER BY CustomerID
  '''

# SQLを実行して、結果をSpark Dataframeで受ける
metrics_sql = spark.sql(sql)

# 結果を確認する
display(metrics_sql)  

# COMMAND ----------

# DBTITLE 1,Python(Spark SQL API)による算出
from pyspark.sql.functions import to_date, datediff, max, min, countDistinct, count, sum, when
from pyspark.sql.types import *

# 有効な顧客注文を含むレコードのみを抜き出す
x = (
    orders
      .where(orders.CustomerID.isNotNull())
      .withColumn('transaction_at', to_date(orders.InvoiceDate))
      .groupBy(orders.CustomerID, 'transaction_at')
      .agg(sum(orders.SalesAmount).alias('salesamount'))   # SALES AMOUNT
    )

# 最後のトランザクション発生日を取得する
y = (
  orders
    .groupBy()
    .agg(max(to_date(orders.InvoiceDate)).alias('current_dt'))
  )


# 顧客毎の最初のトランザクション日時を算出
z = (
  orders
    .groupBy(orders.CustomerID)
    .agg(min(to_date(orders.InvoiceDate)).alias('first_at'))
  )


# 顧客の購入履歴を日時情報と結合させる
a = (x
    .crossJoin(y)
    .join(z, x.CustomerID==z.CustomerID, how='inner')
    .select(
      x.CustomerID.alias('customerid'), 
      z.first_at, 
      x.transaction_at,
      x.salesamount,               # SALES AMOUNT
      y.current_dt
      )
    )

# 顧客毎に関連するメトリックを算出する
metrics_api = (a
           .groupBy(a.customerid, a.current_dt, a.first_at)
           .agg(
             (countDistinct(a.transaction_at)-1).cast(FloatType()).alias('frequency'),
             datediff(max(a.transaction_at), a.first_at).cast(FloatType()).alias('recency'),
             datediff(a.current_dt, a.first_at).cast(FloatType()).alias('T'),
             when(countDistinct(a.transaction_at)==1,0)                           # MONETARY VALUE
               .otherwise(
                 sum(
                   when(a.first_at==a.transaction_at,0)
                     .otherwise(a.salesamount)
                   )/(countDistinct(a.transaction_at)-1)
                 ).alias('monetary_value')
               )
           .select('customerid','frequency','recency','T','monetary_value')
           .orderBy('customerid')
          )

# 結果の確認
display(metrics_api)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 以前のように、いくつかの要約統計を使って、SQLで生成されたデータセットとlifetimesライブラリで生成されたデータセットが同じであることを確認していきましょう:

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリから算出した結果のサマリ統計
# summary data from lifetimes
metrics_pd.describe()

# COMMAND ----------

# DBTITLE 1,SQLから算出した結果のサマリ統計
# summary data from SQL statement
metrics_sql.toPandas().describe()

# COMMAND ----------

# DBTITLE 1,Python(Spark SQL API)で算出した結果のサマリ統計
# summary data from pyspark.sql API
metrics_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Eこの計算を発展させて、キャリブレーション期間とホールドアウト期間の値を導き出すと、以下のようなロジックになります。:
# MAGIC 
# MAGIC 注：ここでもウィジェットを使ってホールドアウト期間の日数を定義しています。.

# COMMAND ----------

# ホールドアウト期間を指定するためのウィジットを定義(デフォルト: 90日)
dbutils.widgets.text('holdout days', '90')

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリによる算出
from datetime import timedelta

# 最後のトランザクション発生日をデータセットのエンドポイント(=「今日」)と見なす。
current_date = orders_pd['InvoiceDate'].max()

# キャリブレーション期間の最終日を算出
holdout_days = int(dbutils.widgets.get('holdout days'))
calibration_end_date = current_date - timedelta(days = holdout_days)

# 必要な顧客メトリックを算出する
metrics_cal_pd = (
  lifetimes.utils.calibration_and_holdout_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date,
    calibration_period_end=calibration_end_date,
    freq='D',
    monetary_value_col='SalesAmount'  # use sales amount to determine monetary value
    )
  )

# 結果を数行表示して確認
metrics_cal_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC SQLおよびPython(Spark SQL API)での実装は以下のとおりです。

# COMMAND ----------

# DBTITLE 1,SQLにより算出
sql = '''
WITH CustomerHistory 
  AS (
    SELECT  -- nesting req'ed b/c can't SELECT DISTINCT on widget parameter
      m.*,
      getArgument('holdout days') as duration_holdout
    FROM (
      SELECT
        x.customerid,
        z.first_at,
        x.transaction_at,
        y.current_dt,
        x.salesamount
      FROM (                                            -- CUSTOMER DAILY SUMMARY
        SELECT 
          customerid, 
          TO_DATE(invoicedate) as transaction_at, 
          SUM(SalesAmount) as salesamount 
        FROM orders 
        GROUP BY customerid, TO_DATE(invoicedate)
        ) x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid is not null
      ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(a.monetary_value,0.0) as monetary_value_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    COALESCE(b.monetary_value_holdout, 0.0) as monetary_value_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, int(p.duration_holdout) ), p.first_at) as float) as T,
        CASE                                              -- MONETARY VALUE CALCULATION
          WHEN COUNT(DISTINCT p.transaction_at)=1 THEN 0    -- 0 if only one order
          ELSE
            SUM(
              CASE WHEN p.first_at=p.transaction_at THEN 0  -- daily average of all but first order
              ELSE p.salesamount
              END
              ) / (COUNT(DISTINCT p.transaction_at)-1)
          END as monetary_value    
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB( p.current_dt, int(p.duration_holdout) )   -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.duration_holdout, p.current_dt, p.first_at
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT p.transaction_at) as float) as frequency_holdout,
    AVG(p.salesamount) as monetary_value_holdout      -- MONETARY VALUE CALCULATION
  FROM CustomerHistory p
  WHERE 
    p.transaction_at >= DATE_SUB(p.current_dt, int(p.duration_holdout) ) AND  -- LIMIT THIS QUERY TO DATA IN THE HOLDOUT PERIOD
    p.transaction_at <= p.current_dt
  GROUP BY p.customerid
  ) b
  ON a.customerid=b.customerid
ORDER BY CustomerID
'''

metrics_cal_sql = spark.sql(sql)
display(metrics_cal_sql)

# COMMAND ----------

# DBTITLE 1,Python(Spark SQL API)を使って算出
from pyspark.sql.functions import avg, date_sub, coalesce, lit, expr

# valid customer orders
x = (
  orders
    .where(orders.CustomerID.isNotNull())
    .withColumn('transaction_at', to_date(orders.InvoiceDate))
    .groupBy(orders.CustomerID, 'transaction_at')
    .agg(sum(orders.SalesAmount).alias('salesamount'))
  )

# calculate last date in dataset
y = (
  orders
    .groupBy()
    .agg(max(to_date(orders.InvoiceDate)).alias('current_dt'))
  )

# calculate first transaction date by customer
z = (
  orders
    .groupBy(orders.CustomerID)
    .agg(min(to_date(orders.InvoiceDate)).alias('first_at'))
  )

# combine customer history with date info (CUSTOMER HISTORY)
p = (x
    .crossJoin(y)
    .join(z, x.CustomerID==z.CustomerID, how='inner')
    .withColumn('duration_holdout', lit(int(dbutils.widgets.get('holdout days'))))
    .select(
      x.CustomerID.alias('customerid'),
      z.first_at, 
      x.transaction_at, 
      y.current_dt, 
      x.salesamount,
      'duration_holdout'
      )
     .distinct()
    )

# calculate relevant metrics by customer
# note: date_sub requires a single integer value unless employed within an expr() call
a = (p
       .where(p.transaction_at < expr('date_sub(current_dt, duration_holdout)')) 
       .groupBy(p.customerid, p.current_dt, p.duration_holdout, p.first_at)
       .agg(
         (countDistinct(p.transaction_at)-1).cast(FloatType()).alias('frequency_cal'),
         datediff( max(p.transaction_at), p.first_at).cast(FloatType()).alias('recency_cal'),
         datediff( expr('date_sub(current_dt, duration_holdout)'), p.first_at).cast(FloatType()).alias('T_cal'),
         when(countDistinct(p.transaction_at)==1,0)
           .otherwise(
             sum(
               when(p.first_at==p.transaction_at,0)
                 .otherwise(p.salesamount)
               )/(countDistinct(p.transaction_at)-1)
             ).alias('monetary_value_cal')
       )
    )

b = (p
      .where((p.transaction_at >= expr('date_sub(current_dt, duration_holdout)')) & (p.transaction_at <= p.current_dt) )
      .groupBy(p.customerid)
      .agg(
        countDistinct(p.transaction_at).cast(FloatType()).alias('frequency_holdout'),
        avg(p.salesamount).alias('monetary_value_holdout')
        )
   )

metrics_cal_api = (
                 a
                 .join(b, a.customerid==b.customerid, how='left')
                 .select(
                   a.customerid.alias('CustomerID'),
                   a.frequency_cal,
                   a.recency_cal,
                   a.T_cal,
                   a.monetary_value_cal,
                   coalesce(b.frequency_holdout, lit(0.0)).alias('frequency_holdout'),
                   coalesce(b.monetary_value_holdout, lit(0.0)).alias('monetary_value_holdout'),
                   a.duration_holdout
                   )
                 .orderBy('CustomerID')
              )

display(metrics_cal_api)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC それぞれの結果を比較する

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリから算出した結果のサマリ統計
# summary data from lifetimes
metrics_cal_pd.describe()

# COMMAND ----------

# DBTITLE 1,SQLから算出した結果のサマリ統計
# summary data from SQL statement
metrics_cal_sql.toPandas().describe()

# COMMAND ----------

# DBTITLE 1,Python(Spark SQL API)から算出した結果のサマリ統計
# summary data from pyspark.sql API
metrics_cal_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC lifetimesライブラリで計算された金銭的なホールドアウト値を注意深く見てみましょう。算出された値は、Sparkコードで算出された値よりもかなり低いことに気づくはずです。これは、lifetimesライブラリが、取引日の合計を平均する代わりに、特定の取引日の個々のラインアイテムを平均しているためです。 lifetimesライブラリの管理者にコードの変更のリクエストをしているのですが、まだ未対応のままになっています。ここでは取引日合計の平均が正しいと思われるので、このノートブックの残りの部分ではそれを使用します。
# MAGIC 
# MAGIC lifetimesライブラリで生成される値と同じ値をSparkで生成したい場合は、以下の2つのセルを参照ください。
# MAGIC ここでSQLを使用してlifetimesライブラリのロジックを再現してあります。

# COMMAND ----------

# DBTITLE 1,SQLを用いて、lifetimeライブラリのmonetary_holdout値を算出
sql = '''
WITH CustomerHistory 
  AS (
    SELECT  -- nesting req'ed b/c can't SELECT DISTINCT on widget parameter
      m.*,
      getArgument('holdout days') as duration_holdout
    FROM (
      SELECT
        x.customerid,
        z.first_at,
        x.transaction_at,
        y.current_dt,
        x.salesamount
      FROM (                                            -- CUSTOMER DAILY SUMMARY
        SELECT 
          customerid, 
          TO_DATE(invoicedate) as transaction_at, 
          SUM(SalesAmount) as salesamount 
        FROM orders 
        GROUP BY customerid, TO_DATE(invoicedate)
        ) x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid is not null
      ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(a.monetary_value,0.0) as monetary_value_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    COALESCE(b.monetary_value_holdout, 0.0) as monetary_value_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, int(p.duration_holdout) ), p.first_at) as float) as T,
        CASE                                              -- MONETARY VALUE CALCULATION
          WHEN COUNT(DISTINCT p.transaction_at)=1 THEN 0    -- 0 if only one order
          ELSE
            SUM(
              CASE WHEN p.first_at=p.transaction_at THEN 0  -- daily average of all but first order
              ELSE p.salesamount
              END
              ) / (COUNT(DISTINCT p.transaction_at)-1)
          END as monetary_value    
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB(p.current_dt, int( p.duration_holdout) )  -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.duration_holdout, p.current_dt, p.first_at
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT TO_DATE(p.invoicedate)) as float) as frequency_holdout,
    AVG(p.salesamount) as monetary_value_holdout      -- MONETARY VALUE CALCULATION
  FROM orders p
  CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) q                                -- current date (according to dataset)
  INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) r  -- first order per customer
    ON p.customerid=r.customerid
  WHERE 
    p.customerid is not null AND
    TO_DATE(p.invoicedate) >= DATE_SUB(q.current_dt, int( getArgument('holdout days') ) ) AND  -- LIMIT THIS QUERY TO DATA IN THE HOLDOUT PERIOD
    TO_DATE(p.invoicedate) <= q.current_dt
  GROUP BY p.customerid
  ) b
  ON a.customerid=b.customerid
ORDER BY CustomerID
'''

metrics_cal_sql_alt = spark.sql(sql)
display(metrics_cal_sql_alt)

# COMMAND ----------

# lifetimesライブラリによる算出結果の統計サマリ (比較用)
metrics_cal_pd.describe()

# COMMAND ----------

# SQLによる算出結果の統計サマリ
# "monetary_value_holdout"はlifetmesライブラリの実装と同じ仕様にしている
metrics_cal_sql_alt.toPandas().describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC これ以降は、前回のノートブックと同様に、リピート購入がある顧客に限定して分析を行います。

# COMMAND ----------

# リピート購入のない顧客を除外する (全データセット対象)
filtered = metrics_api.where(metrics_api.frequency > 0)

# リピート購入のない顧客を除外する (キャリブレーション期間を対象)
filtered_cal = metrics_cal_api.where(metrics_cal_api.frequency_cal > 0)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 最後に、今回のデータセットに含まれる1日の合計値がマイナスになっているレコードに関しての考慮が必要になります。 
# MAGIC このデータセットの元となった小売業者についての文脈情報がなければ、これらのマイナス値は返品されたものだと考えられます。
# MAGIC 
# MAGIC 理想的には、返品された商品を元の購入商品と照合し、元の取引日に合わせて金額を調整することです。しかし、これを一貫して行うために必要な情報を持っていないため、マイナスのリターン値を毎日の取引合計に単純に含めることにします。これにより、1日の合計が£0以下になる場合は、その値を分析から除外します。実証実験の場以外では、これは一般的に適切ではありません。

# COMMAND ----------

# exclude dates with negative totals (see note above) 
filtered = filtered.where(filtered.monetary_value > 0)
filtered_cal = filtered_cal.where(filtered_cal.monetary_value_cal > 0)

# COMMAND ----------

# MAGIC %md ###Step 4: 頻度と金銭的価値の独立性を検証する
# MAGIC 
# MAGIC モデル化を進める前に、ここで採用するガンマ・ガンマモデル(前述の2つのガンマ分布にちなんで命名されている)は、顧客の購入頻度がその購入金額に影響しないことを前提としています。 これを検証することは重要で、頻度と金額の測定基準に対する単純なピアソン係数を計算することで可能になります。 今回の分析では、キャリブレーションとホールドアウトのサブセットを無視して、データセット全体に対してこれを行います。

# COMMAND ----------

# 相関係数(ピアソン係数)を算出
filtered.corr('frequency', 'monetary_value')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 完全に独立しているわけではありませんが、この2つの値の相関はかなり低いので、モデルのトレーニングを進めても問題ないと言えるでしょう。 

# COMMAND ----------

# MAGIC %md ###Step 5: 消費モデルを訓練する
# MAGIC 
# MAGIC 測定基準が確立されたので、将来のトランザクションイベントから得られる金銭的価値を推定するモデルを訓練することができます。ここで使用するモデルは、[Gamma-Gamma model](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)と呼ばれ、顧客集団の支出分布から得られるガンマ分布のパラメータに対して、個々の顧客の支出のガンマ分布を適合させるものです。計算は複雑ですが、lifetimesライブラリを用いて容易に算出できます。
# MAGIC 
# MAGIC そのためにはまず、モデルが使用するL2正則化パラメータの最適な値を決定する必要があります。 そこで、前のノートブックと同様に[hyperopt](http://hyperopt.github.io/hyperopt/)を使って、効果的にパラメータ探索を行います。

# COMMAND ----------

from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, space_eval

from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter

# サーチスペース(探索範囲)を定義
search_space = hp.uniform('l2', 0.0, 1.0)

# 評価関数を定義
def score_model(actuals, predicted, metric='mse'):
  # メトリック名は小文字に揃える
  metric = metric.lower()
  
  # 平均二乗誤差(MSE)と平均平方二乗誤差(RMSE)の場合
  if metric=='mse' or metric=='rmse':
    val = np.sum(np.square(actuals-predicted))/actuals.shape[0]
    if metric=='rmse':
        val = np.sqrt(val)
  
  # 平均絶対誤差(MAE)の場合
  elif metric=='mae':
    np.sum(np.abs(actuals-predicted))/actuals.shape[0]
  
  # その他の場合
  else:
    val = None
  
  return val


# モデルトレーニングおよび評価の関数を定義する
def evaluate_model(param):
  
  # "input_pd"データフレームのレプリカを用意
  data = inputs.value
  
  # 入力パラメータの抽出
  l2_reg = param
  
  # Gramma-Gamma-Filterモデルのインスタンス化
  model = GammaGammaFitter(penalizer_coef=l2_reg)
  
  # モデルのフィッティング(トレーニング)
  model.fit(data['frequency_cal'], data['monetary_value_cal'])
  
  # モデルの評価
  monetary_actual = data['monetary_value_holdout']
  monetary_predicted = model.conditional_expected_average_profit(data['frequency_holdout'], data['monetary_value_holdout'])
  mse = score_model(monetary_actual, monetary_predicted, 'mse')
  
  # スコアとステータスを戻り値として返す
  return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# Hyperoptの並列実行環境としてSparkのworkerを使用するように設定
spark_trials = SparkTrials(parallelism=8)

# Hpyeroptのパラメータ探索アルゴリズムの設定(今回はTPEを使用する)
algo = tpe.suggest

# "input_pd"データフレームのコピーを各workerに配っておく
input_pd = filtered_cal.where(filtered_cal.monetary_value_cal > 0).toPandas()
inputs = sc.broadcast(input_pd)

# Hyper-parameter Tuningを実行 (かつ、MLflowでトラッキングする)
argmin = fmin(
  fn=evaluate_model,
  space=search_space,
  algo=algo,
  max_evals=100,
  trials=spark_trials
  )

# Broadcastしたデータをリリースする
inputs.unpersist()

# COMMAND ----------

# 最適なハイパーパラメータを表示
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 最適なL2値がわかったところで、最終的な消費モデルを作成してみましょう。

# COMMAND ----------

# ハイパーパラメータを取得
l2_reg = space_eval(search_space, argmin)

# 上記のハイパーパラメータを使って、モデルのインスタンス化
spend_model = GammaGammaFitter(penalizer_coef=l2_reg)

# モデルのトレーニング
spend_model.fit(input_pd['frequency_cal'], input_pd['monetary_value_cal'])

# COMMAND ----------

# MAGIC %md ###Step 6: 消費モデルの評価
# MAGIC 
# MAGIC モデルの評価はとてもシンプルです。 予測値とホールドアウト期間の実績がどの程度一致しているかを調べ、そこからMSEを算出します。

# COMMAND ----------

# モデルの評価
monetary_actual = input_pd['monetary_value_holdout']
monetary_predicted = spend_model.conditional_expected_average_profit(input_pd['frequency_holdout'], input_pd['monetary_value_holdout'])
mse = score_model(monetary_actual, monetary_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC また、Gamma-Gammaモデルを説明した[原著論文](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)で採用された手法である、予測された支出額と実際の支出額がどのように一致しているかを視覚的に確認することもできます。

# COMMAND ----------

import matplotlib.pyplot as plt

# ヒストグラムのbins数を設定
bins = 10

# plot size
plt.figure(figsize=(15, 5))

# histogram plot values and presentation
plt.hist(monetary_actual, bins, label='actual', histtype='bar', color='STEELBLUE', rwidth=0.99)
plt.hist( monetary_predicted, bins, label='predict', histtype='step', color='ORANGE',  rwidth=0.99)

# place legend on chart
plt.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 10個のbins数でのヒストグラムでは、モデルは実際のデータとうまく一致しているように見えます。
# MAGIC ビンの数を増やすと、モデルはデータの残りの構造に従う一方で、最も価値の低い消費の発生を過小評価していることがわかります。
# MAGIC 興味深いことに、先に引用した論文でも同じようなパターンが観察されています。

# COMMAND ----------

# ヒストグラムのbins数を40に増やす
bins = 40

# plot size
plt.figure(figsize=(15, 5))

# histogram plot values and presentation
plt.hist(monetary_actual, bins, label='actual', histtype='bar', color='STEELBLUE', rwidth=0.99)
plt.hist( monetary_predicted, bins, label='predict', histtype='step', color='ORANGE',  rwidth=0.99)

# place legend on chart
plt.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md ###Step 7: 顧客生涯価値を算出する
# MAGIC 
# MAGIC 消費モデルでは、将来の購入イベントから得られる可能性のある金銭的価値を計算することができます。 
# MAGIC 将来の支出イベントの可能性を計算するライフタイムモデルと組み合わせて使用することで、将来の期間における顧客生涯価値を導き出すことができます。
# MAGIC 
# MAGIC これを実証するには、まずライフタイムモデルをトレーニングする必要があります。 ここでは、BG/NBDモデルに、以前のノートブックの実行時に得られたL2パラメータ設定を使用します。

# COMMAND ----------

from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

# Spark-DFからPandas-DFに変換する
lifetime_input_pd = filtered_cal.toPandas() 

# モデルのインスタンス作成(前回のNoteookのHyperparamチューニングの結果からパラメータを設定する)
lifetimes_model = BetaGeoFitter(penalizer_coef=0.9995179967263891)

# モデルのトレーニング
lifetimes_model.fit(lifetime_input_pd['frequency_cal'], lifetime_input_pd['recency_cal'], lifetime_input_pd['T_cal'])

# スコアリング
frequency_holdout_actual = lifetime_input_pd['frequency_holdout']
frequency_holdout_predicted = lifetimes_model.predict(lifetime_input_pd['duration_holdout'], lifetime_input_pd['frequency_cal'], lifetime_input_pd['recency_cal'], lifetime_input_pd['T_cal'])
mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC では、これらを組み合わせてCLVを計算してみましょう。ここでは、毎月の割引率を1％として、12ヶ月間のCLVを計算します。
# MAGIC 
# MAGIC 注：CFOは通常、この種の計算に使用すべき割引率を定義します。 割引率が月単位の割引率であることを確認してください。 年単位の割引率が提供されている場合は、必ず[この式](https://www.experiglot.com/2006/06/07/how-to-convert-from-an-annual-rate-to-an-effective-periodic-rate-javascript-calculator/)を使って月単位に変換してください。

# COMMAND ----------

clv_input_pd = filtered.toPandas()

# 1年間のCLVを顧客毎に算出する
clv_input_pd['clv'] = (
  spend_model.customer_lifetime_value(
    lifetimes_model, #the model to use to predict the number of future transactions
    clv_input_pd['frequency'],
    clv_input_pd['recency'],
    clv_input_pd['T'],
    clv_input_pd['monetary_value'],
    time=12, # months
    discount_rate=0.01 # monthly discount rate ~ 12.7% annually
  )
)

clv_input_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC CLVは、企業がターゲットを絞ったプロモーション活動を計画したり、カスタマー・エクイティを評価したりする際に使用される強力な指標です。そのため、私たちのモデルを使いやすい関数に変換して、バッチ、ストリーミング、インタラクティブなシナリオで使用できるようにすることができれば、非常に便利になります。
# MAGIC 
# MAGIC 前回のノートをご覧になった方は、私たちがどこに向かっているのかご存知でしょう。 ここでは、CLVの計算が1つのモデルではなく、2つのモデルに依存していることを指摘しておきます。 問題はありません。 ここでは、生涯モデルを消費モデルに関連するピクルス化されたアーティファクトとして保存し、消費モデル用に開発するカスタムラッパーで、生涯モデルを再インスタンス化して、予測に利用できるようにします。
# MAGIC 
# MAGIC まずは、ライフタイムモデルを一時的に保存してみましょう。:

# COMMAND ----------

# lifetimesモデルを保存するテンポラリなパスを設定
lifetimes_model_path = '/dbfs/tmp/lifetimes_model.pkl'

# 以前の結果があれば削除する
try:
  dbutils.fs.rm(lifetimes_model_path)
except:
  pass

# 保存する
lifetimes_model.save_model(lifetimes_model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC それでは、消費モデルのカスタムラッパーを定義してみましょう。 ここで、`predict()` メソッドは非常にシンプルで、CLV値を返すだけです。 また、月の値と割引率が入力データに含まれていることを前提としています。
# MAGIC 
# MAGIC `Predict()` メソッドのロジックを変更したほか、`load_context()` の定義を新たに設けました。 このメソッドは、[mlflow](https://mlflow.org/)モデルがインスタンス化されたときに呼び出されます。 ここでは、lifetimeモデルの成果物をロードします。

# COMMAND ----------

import mlflow 
import mlflow.pyfunc

# lifetimesモデルのラッパークラスを作成
class _clvModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, spend_model):
      self.spend_model = spend_model
        
    def load_context(self, context):
      # lifetimesライブラリからBase Model Fitterをimportしておく
      from lifetimes.fitters.base_fitter import BaseFitter
      
      # モデルのインスタンスを作成
      self.lifetimes_model = BaseFitter()
      
      # MLflowからlifetimesモデルをロードする
      self.lifetimes_model.load_model(context.artifacts['lifetimes_model'])
      
    def predict(self, context, dataframe):
      
      # 入力データから各種パラメータを抽出
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]
      monetary_value = dataframe.iloc[:,3]
      months = int(dataframe.iloc[0,4])
      discount_rate = float(dataframe.iloc[0,5])
      
      # CLV推定を実施する
      results = pd.DataFrame(
          self.spend_model.customer_lifetime_value(
            self.lifetimes_model, #the model to use to predict the number of future transactions
            frequency,
            recency,
            T,
            monetary_value,
            time=months,
            discount_rate=discount_rate
            ),
          columns=['clv']
          )
      
      return results[['clv']]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 次に、消費モデルをmlflowに保存します。

# COMMAND ----------

# lifetimesライブラリをconda環境に追加
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][-1]['pip'] += ['lifetimes==0.10.1'] # lifetimesのversionはノートブック前半でinstallしたversionに合わせる

# モデルトレーニング実行をMLflowに保存する
with mlflow.start_run(run_name='deployment run') as run:
  
  # lifetimeモデルをartifact "lifetime_model"としてmlflowでトラックするための準備
  artifacts = {'lifetimes_model': lifetimes_model_path}
  
  # MLflowでトラック
  mlflow.pyfunc.log_model(
    'model', 
    python_model=_clvModelWrapper(spend_model), 
    conda_env=conda_env,
    artifacts=artifacts
    )

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 先ほどと同様に、モデルから関数を作成します。

# COMMAND ----------

# 関数の戻り値のデータ型(スキーマ)を定義
result_schema = DoubleType()

# MLflowに登録されたモデルをベースにした関数を定義する
clv_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'runs:/{0}/model'.format(run.info.run_id), 
  result_type=result_schema
  )

# 上記の関数をSQLで使用するためにUDFとして登録する
_ = spark.udf.register('clv', clv_udf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC これでモデルがPython/SQLで利用可能になりました。

# COMMAND ----------

# 次のセルでSQLを実行するためのtemp viewを作成しておく
filtered.createOrReplaceTempView('customer_metrics')

# Spark DataFrameに関数を適用させる
display(
  filtered
    .withColumn(
      'clv', 
      clv_udf(filtered.frequency, filtered.recency, filtered.T, filtered.monetary_value, lit(12), lit(0.01))
      )
    .selectExpr(
      'customerid', 
      'clv'
      )
  )

# COMMAND ----------

# MAGIC %md It can also be used with SQL:

# COMMAND ----------

# MAGIC %sql -- 顧客生涯価値を算出する
# MAGIC 
# MAGIC SELECT
# MAGIC   customerid,
# MAGIC   clv(
# MAGIC     frequency,
# MAGIC     recency,
# MAGIC     T,
# MAGIC     monetary_value,
# MAGIC     12,
# MAGIC     0.01
# MAGIC     ) as clv
# MAGIC FROM customer_metrics;
