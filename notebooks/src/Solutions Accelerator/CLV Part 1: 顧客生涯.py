# Databricks notebook source
# MAGIC %md ##将来のカスタマーエンゲージメントの確率を算出する
# MAGIC 
# MAGIC サブスクリプション型ではない小売モデルでは、顧客は長期的なコミットメントなしに出入りするため、ある顧客が将来的に戻ってくるかどうかを判断することは非常に困難です。効果的なマーケティングキャンペーンを実施するためには、顧客が再来店する可能性を判断することが重要です。
# MAGIC 
# MAGIC 一度退会した可能性の高いお客様に再び来店していただくためには、異なるメッセージングやプロモーションが必要になる場合があります。また、一度ご来店いただいたお客様には、当社でのご購入の幅や規模を拡大していただくためのマーケティング活動に、より積極的に取り組んでいただける可能性があります。お客様が将来的に関与する可能性についてどのような位置にいるかを理解することは、お客様に合わせたマーケティング活動を行う上で非常に重要です。
# MAGIC 
# MAGIC Peter Faderらが提唱したBTYD（Buy 'til You Die）モデルは、2つの基本的な顧客指標、すなわち、顧客が最後に購入した時の記憶と、顧客の生涯におけるリピート取引の頻度を利用して、将来の再購入の確率を導き出すものです。これは、顧客の履歴を、購入頻度の分布と、前回の購入後のエンゲージメントの低下を表す曲線に当てはめることで行われます。
# MAGIC 
# MAGIC これらのモデルの背後にある数学はかなり複雑ですが、ありがたいことに[lifetimes](https://pypi.org/project/Lifetimes/)ライブラリにまとめられているため、従来の企業でも採用しやすくなっています。このノートの目的は、これらのモデルを顧客の取引履歴にどのように適用するか、またマーケティングプロセスにどのように統合するかを検討することです。

# COMMAND ----------

# MAGIC %md ###Step 1: 環境構築
# MAGIC 
# MAGIC このノートブックを実行するには、Databricks ML Runtime v6.5+以降クラスタに接続する必要があります。このバージョンのDatabricks Runtimeは、多くのライブラリがプリインストールされていますが、このNotebookでは追加で以下のPythonライブラリを使用しているので、別途インストールする必要がります。
# MAGIC 
# MAGIC * xlrd
# MAGIC * lifetimes==0.10.1
# MAGIC * nbconvert
# MAGIC 
# MAGIC インストールの方法は[こちら](https://docs.microsoft.com/ja-jp/azure/databricks/libraries/)を参照ください。PyPIライブラリのソースおよび上記のライブラリを両方使用します。
# MAGIC 
# MAGIC インストールが完了したら、このノートブックをそのクラスタにアタッチしてください。
# MAGIC 
# MAGIC (訳者注: このNotebookではすでにデータ形式が利用しやすいCSV形式になっているので、xlrd, nbconvertはインストール不要になっています。)

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリのインストール
# MAGIC %pip install lifetimes==0.10.1

# COMMAND ----------

# MAGIC %md 
# MAGIC ライブラリをインストールしたら、BTYDモデルを検証するためのサンプルデータセットをロードしてみましょう。ここで使用するデータセットは、UCI Machine Learning Repositoryから入手可能な[Online Retailデータセット](http://archive.ics.uci.edu/ml/datasets/Online+Retail)です。
# MAGIC 
# MAGIC このデータセットは、Microsoft Excelワークブック（XLSX）として提供されています。このXLSXファイルをローカルシステムにダウンロードした後、[ここ](https://docs.databricks.com/data/tables.html#create-table-ui)で紹介する手順に従ってDatabricks環境にロードします。
# MAGIC 
# MAGIC なお、ファイルのインポートを実行する際、インポート処理を完了するために「UIでテーブルを作成」または「ノートブックでテーブルを作成」オプションを選択する必要はありません。また、XLSXファイルの名前にはサポートされていないスペース文字が含まれているため、インポート時にファイル名が変更されます。そのため、インポート処理で割り当てられたファイルの新しい名前をプログラムで見つける必要があります。
# MAGIC 
# MAGIC 以下では、XLSXのアップロード先として`/FileStore/tables/online_retail/`を使用することにします。
# MAGIC 
# MAGIC (注: `/FileStore/tables/online_retail/`配下にあるデータはすでにXLSX形式からCSV形式に変換されているので、上記のxrldライブラリは不要になっている)

# COMMAND ----------

# MAGIC %sh
# MAGIC wget 'https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/Online_Retail-93ac4.csv'
# MAGIC cp Online_Retail-93ac4.csv /dbfs/FileStore/tables/online_retail/

# COMMAND ----------

dbutils.fs.ls('file:///dbfs/FileStore/tables/online_retail')

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/online_retail

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

# xlsxファイルをPandasのDataFrameとして読み込む
# (このため、ライブラリ "xlrd" が必要) 
# orders_pd = pd.read_excel(
#   xlsx_filename, 
#   sheet_name='Online Retail',
#   header=0, # 第一行目はヘッダーになっている
#   dtype=orders_schema
#   )

#　元のファイルがCSVになっているので、そのまま読み込む
orders_pd = pd.read_csv(
  xlsx_filename, 
  sep=',',
  #sheet_name='Online Retail',
  header=0, # 第一行目はヘッダーになっている
  dtype=orders_schema,
  parse_dates=['InvoiceDate']
  )

# 読み込んだDataFrameを確認してみる
orders_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ワークブックのデータは、「Online Retail」スプレッドシートの範囲として整理されています。 各レコードは、販売取引のラインアイテムを表しています。それぞれのフィールドは以下の通りです。
# MAGIC 
# MAGIC | Field | Description |
# MAGIC |-------------:|-----:|
# MAGIC |InvoiceNo|各トランザクションに一意に割り当てられた6桁の整数|
# MAGIC |StockCode|それぞれの製品に一意に割り当てられた5桁の整数|
# MAGIC |Description|プロダクト名(アイテム名)|
# MAGIC |Quantity|トランザクションあたりの数量|
# MAGIC |InvoiceDate|Invoiceの日時(`mm/dd/yy hh:mm`フォーマット)|
# MAGIC |UnitPrice|製品単価 (£, ポンド)|
# MAGIC |CustomerID| 顧客に一意に割り当てられた5桁の整数|
# MAGIC |Country|顧客の居住国|
# MAGIC 
# MAGIC これらのフィールドのうち、今回の作業で特に注目すべきものは、取引を識別するInvoiceNo、取引の日付を識別するInvoiceDate、複数の取引で顧客を一意に識別するCustomerIDです。(別のノートブックでは、UnitPriceフィールドとQuantityフィールドを使って、トランザクションの金銭的価値を調べます)。

# COMMAND ----------

# MAGIC %md ###Step 2: データセットの探索
# MAGIC 
# MAGIC SQLを使ってデータを探索してみましょう。そのために、まずPandas DataFrameをSpark DataFrameに変換し、Temporary Viewを作成します。

# COMMAND ----------

# Pandas-DF から　Spark-DF　へ変換
orders = spark.createDataFrame(orders_pd)

# Temp Viewの作成
orders.createOrReplaceTempView('orders') 

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC データセットの取引状況を見てみると、
# MAGIC 
# MAGIC * 最初の取引が2010年12月1日
# MAGIC * 最後の取引が2011年12月9日
# MAGIC * データセットの期間は1年を少し超えている
# MAGIC 
# MAGIC ことがわかります。
# MAGIC 
# MAGIC また、1日のトランザクション数を見ると、このオンライン小売業者の1日のアクティビティにはかなりのボラティリティがあることがわかります。

# COMMAND ----------

# DBTITLE 1,1日のトランザクション数
# MAGIC %sql -- 1日ごとのユニークなトランザクション数
# MAGIC 
# MAGIC SELECT 
# MAGIC   TO_DATE(InvoiceDate) as InvoiceDate,
# MAGIC   COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC GROUP BY TO_DATE(InvoiceDate)
# MAGIC ORDER BY InvoiceDate;

# COMMAND ----------

# MAGIC %md
# MAGIC 月別の活動を要約することで、この現象を少し滑らかにすることができます。2011年12月は9日間しかなかったので、先月の売上減少のグラフはほとんど無視してよいでしょう。
# MAGIC 
# MAGIC 注：以下の結果セットの背後にあるSQLは、見やすいように隠しておきます。このコードを表示するには、以下の各グラフの上にある **"Show code"** をクリック

# COMMAND ----------

# DBTITLE 1,月次のトランザクション数
# MAGIC %sql -- 月ごとのユニークなトランザクション数
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(InvoiceDate, 'month') as InvoiceMonth,
# MAGIC   COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC GROUP BY TRUNC(InvoiceDate, 'month') 
# MAGIC ORDER BY InvoiceMonth;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC データのある1年強の期間では、4,000人以上のユニークなお客様がいます。
# MAGIC これらのお客様が生み出したユニークな取引は約2万2千件です。

# COMMAND ----------

# DBTITLE 1,顧客数と取引回数の総計
# MAGIC %sql -- unique customers and transactions
# MAGIC 
# MAGIC SELECT
# MAGIC  COUNT(DISTINCT CustomerID) as Customers,
# MAGIC  COUNT(DISTINCT InvoiceNo) as Transactions
# MAGIC FROM orders
# MAGIC WHERE CustomerID IS NOT NULL;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 簡単な割り算で、顧客一人当たり、平均して約5件の取引を担当していることがわかります。しかし、これでは顧客の活動を正確に表すことはできません。
# MAGIC 
# MAGIC 単純平均ではなく、顧客ごとにユニークな取引をカウントし、その値の頻度を調べてみると、多くのお客様が1回の取引を行っていることがわかります。リピート購入回数の分布は、リピート購入回数の分布は、そこから負の二項分布（ほとんどのBTYDモデルの名前に含まれるNBDの頭文字の根拠となっています）と言えるような形で減少していきます。

# COMMAND ----------

# DBTITLE 1,各顧客の取引回数のヒストグラム(分布)
# MAGIC %sql -- the distribution of per-customer transaction counts 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.Transactions,
# MAGIC   COUNT(x.*) as Occurrences
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     COUNT(DISTINCT InvoiceNo) as Transactions 
# MAGIC   FROM orders
# MAGIC   WHERE CustomerID IS NOT NULL
# MAGIC   GROUP BY CustomerID
# MAGIC   ) x
# MAGIC GROUP BY 
# MAGIC   x.Transactions
# MAGIC ORDER BY
# MAGIC   x.Transactions;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 先の分析で、同じ日に発生した顧客の取引を1つの取引にまとめるように変更した場合（このパターンは後に計算する指標と一致します）、より多くの顧客が非リピート顧客として識別されますが、全体的なパターンは変わりません。

# COMMAND ----------

# MAGIC %sql -- the distribution of per-customer transaction counts
# MAGIC      -- with consideration of same-day transactions as a single transaction 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.Transactions,
# MAGIC   COUNT(x.*) as Occurances
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     COUNT(DISTINCT TO_DATE(InvoiceDate)) as Transactions
# MAGIC   FROM orders
# MAGIC   WHERE CustomerID IS NOT NULL
# MAGIC   GROUP BY CustomerID
# MAGIC   ) x
# MAGIC GROUP BY 
# MAGIC   x.Transactions
# MAGIC ORDER BY
# MAGIC   x.Transactions;

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC リピート購入のお客様に焦点を当てて、購入イベント間の日数の分布を調べることができます。ここで重要なのは、ほとんどのお客様が購入後2～3ヶ月以内にサイトに戻ってきていることです。それ以上の期間が空くこともありますが、再購入までの期間が長いお客様はかなり少ないです。
# MAGIC 
# MAGIC このことは、BYTDモデルの文脈で理解することが重要です。つまり、最後にお客様にお会いしてからの時間は、お客様が再来店されるかどうかを判断するための重要な要素であり、お客様が最後に購入されてからの時間が長くなればなるほど、再来店の確率は下がっていきます。

# COMMAND ----------

# DBTITLE 1,リピート時の取引の間隔(日数)
# MAGIC %sql -- distribution of per-customer average number of days between purchase events
# MAGIC 
# MAGIC WITH CustomerPurchaseDates
# MAGIC   AS (
# MAGIC     SELECT DISTINCT
# MAGIC       CustomerID,
# MAGIC       TO_DATE(InvoiceDate) as InvoiceDate
# MAGIC     FROM orders 
# MAGIC     WHERE CustomerId IS NOT NULL
# MAGIC     )
# MAGIC SELECT -- Per-Customer Average Days Between Purchase Events
# MAGIC   AVG(
# MAGIC     DATEDIFF(a.NextInvoiceDate, a.InvoiceDate)
# MAGIC     ) as AvgDaysBetween
# MAGIC FROM ( -- Purchase Event and Next Purchase Event by Customer
# MAGIC   SELECT 
# MAGIC     x.CustomerID,
# MAGIC     x.InvoiceDate,
# MAGIC     MIN(y.InvoiceDate) as NextInvoiceDate
# MAGIC   FROM CustomerPurchaseDates x
# MAGIC   INNER JOIN CustomerPurchaseDates y
# MAGIC     ON x.CustomerID=y.CustomerID AND x.InvoiceDate < y.InvoiceDate
# MAGIC   GROUP BY 
# MAGIC     x.CustomerID,
# MAGIC     x.InvoiceDate
# MAGIC     ) a
# MAGIC GROUP BY CustomerID

# COMMAND ----------

# MAGIC %md ###Step 3: 顧客のメトリックを計算する
# MAGIC 
# MAGIC 私たちが扱うデータセットは、生の取引履歴で構成されています。BTYDモデルを適用するためには、いくつかの顧客ごとの評価指標を導き出す必要があります。</p>
# MAGIC 
# MAGIC * **Frequency** - 観測期間中の取引(買い物)回数。ただし、初回購入は除く。つまり、(全取引回数 - 1)。日毎にカウント。つまり、同日に複数回取引があっても1回とカウントする。
# MAGIC * **Age (T)** - 経過日数, 初めての取引発生した日から現在の日付（またはデータセットの最終の日)
# MAGIC * **Recency** - 直近の取引があった時点のAge。つまり、初回の取引の日から直近(最後の)取引があった日までの経過日数。
# MAGIC 
# MAGIC 顧客年齢などの指標を計算する際には、データセットがいつ終了するかを考慮する必要があることに注意してください。今日の日付を基準にしてこれらのメトリクスを計算すると、誤った結果になる可能性があります。そこで、データセットの最後の日付を特定し、それを *今日の日付* と定義して、すべての計算を行うことにします。
# MAGIC 
# MAGIC ここでは、lifetimesライブラリに組み込まれた機能を使って、どのように計算を行うかをみていきます。

# COMMAND ----------

import lifetimes

# set the last transaction date as the end point for this historical dataset
current_date = orders_pd['InvoiceDate'].max()

# calculate the required customer metrics
metrics_pd = (
  lifetimes.utils.summary_data_from_transaction_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date, 
    freq='D'
    )
  )

# display first few rows
metrics_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC lifetimesライブラリは、多くのPythonライブラリと同様に、シングルスレッドです。このライブラリを使用して大規模なトランザクションデータセットの顧客メトリクスを導出すると、システムを圧迫したり、単に完了までに時間がかかりすぎたりする可能性があります。そこで、Apache Sparkの分散機能を利用してこれらの指標を算出する方法を検討してみましょう。
# MAGIC 
# MAGIC 
# MAGIC 複雑なデータ操作にはSQLが使われることが多いので、まずはSparkのSQL文を使ってみます。
# MAGIC 
# MAGIC このステートメントでは、まず各顧客の注文履歴を、
# MAGIC 1. 顧客ID
# MAGIC 1. 最初の購入日（first_at）
# MAGIC 1. 購入が確認された日（transaction_at）
# MAGIC 1. 現在の日付（この値にはデータセットの最後の日付を使用）
# MAGIC 
# MAGIC で構成します。
# MAGIC 
# MAGIC この履歴から、顧客ごとに、
# MAGIC 1. 繰り返し取引された日の数（frequency）
# MAGIC 1. 最後の取引日から最初の取引日までの日数（recency）
# MAGIC 1. 現在の日付から最初の取引までの日数（T）
# MAGIC 
# MAGIC をカウントすることができます。

# COMMAND ----------

# sql statement to derive summary customer stats
sql = '''
  SELECT
    a.customerid as CustomerID,
    CAST(COUNT(DISTINCT a.transaction_at) - 1 as float) as frequency,
    CAST(DATEDIFF(MAX(a.transaction_at), a.first_at) as float) as recency,
    CAST(DATEDIFF(a.current_dt, a.first_at) as float) as T
  FROM ( -- customer order history
    SELECT DISTINCT
      x.customerid,
      z.first_at,
      TO_DATE(x.invoicedate) as transaction_at,
      y.current_dt
    FROM orders x
    CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
    INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
      ON x.customerid=z.customerid
    WHERE x.customerid IS NOT NULL
    ) a
  GROUP BY a.customerid, a.current_dt, a.first_at
  ORDER BY CustomerID
  '''

# capture stats in dataframe 
metrics_sql = spark.sql(sql)

# display stats
display(metrics_sql)  

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC もちろん，Spark SQLでは，DataFrameをSQL文のみでアクセスする必要はありません。
# MAGIC 
# MAGIC データサイエンティストの好みに合わせて、プログラマティックSQL APIを使って同じ結果を導き出すこともできます。次のセルのコードは、比較のために、前のSQL文の構造を反映するように意図的に組み立てられています。

# COMMAND ----------

from pyspark.sql.functions import to_date, datediff, max, min, countDistinct, count, sum, when
from pyspark.sql.types import *

# valid customer orders
x = orders.where(orders.CustomerID.isNotNull())

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

# combine customer history with date info 
a = (x
    .crossJoin(y)
    .join(z, x.CustomerID==z.CustomerID, how='inner')
    .select(
      x.CustomerID.alias('customerid'), 
      z.first_at, 
      to_date(x.InvoiceDate).alias('transaction_at'), 
      y.current_dt
      )
     .distinct()
    )

# calculate relevant metrics by customer
metrics_api = (a
           .groupBy(a.customerid, a.current_dt, a.first_at)
           .agg(
             (countDistinct(a.transaction_at)-1).cast(FloatType()).alias('frequency'),
             datediff(max(a.transaction_at), a.first_at).cast(FloatType()).alias('recency'),
             datediff(a.current_dt, a.first_at).cast(FloatType()).alias('T')
             )
           .select('customerid','frequency','recency','T')
           .orderBy('customerid')
          )

display(metrics_api)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 結果が同じであることを確認するために、これらの異なるメトリクス・データセットのデータを比較してみましょう。レコードごとに行うのではなく、各データセットの要約統計を計算して、その一貫性を確認してみましょう。
# MAGIC 
# MAGIC 
# MAGIC 注：平均値と標準偏差は、小数点以下10万分の1と100万分の1でわずかに異なることに気づくかもしれません。これは、pandasとSparkのデータフレームのデータ型の違いによるものですが、結果には大きな影響はありません。

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリを使って算出
# summary data from lifetimes
metrics_pd.describe()

# COMMAND ----------

# DBTITLE 1,SQL(Spark SQL)を使って算出
# summary data from SQL statement
metrics_sql.toPandas().describe()

# COMMAND ----------

# DBTITLE 1,Python(pyspark)を使って算出
# summary data from pyspark.sql API
metrics_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ここまでに算出したメトリクスは、時系列データのサマリーを表しています。
# MAGIC 
# MAGIC モデル検証およびオーバーフィッティング回避を考える必要がります。時系列データの一般的なパターンは、時系列の初期部分（校正期間と呼ばれる）でモデルをトレーニングし、時系列の後期部分（ホールドアウト期間と呼ばれる）で検証することです。
# MAGIC 
# MAGIC lifetimesライブラリでは、キャリブレーション期間とホールドアウト期間を用いた顧客ごとのメトリクスの導出がシンプルな関数呼び出しで可能です。
# MAGIC 
# MAGIC 今回のデータセットは限られた範囲のデータで構成されているため、ホールドアウト期間として過去90日分のデータを使用するよう、ライブラリの関数に与えています。この設定の構成を簡単に変更できるように、Databricksではウィジェットと呼ばれるシンプルなパラメータUIが実装されています。
# MAGIC 
# MAGIC 注：次のセルを実行するとノートブックの最上部に、テキストボックス・ウィジェットが追加されます。これを使って、保留期間の日数を変更することができます。

# COMMAND ----------

# NotebookのWidgetを作成 (デフォルト: 90-days)
dbutils.widgets.text('holdout days', '90')

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリを使って算出
from datetime import timedelta

# データセットの最後の日付を分析の「今日の日付」にする
current_date = orders_pd['InvoiceDate'].max()

# キャブレーション期間の最終日を定義する
holdout_days = int(dbutils.widgets.get('holdout days'))
calibration_end_date = current_date - timedelta(days = holdout_days)

# lifetimesライブラリを使って、顧客メトリックを算出する
metrics_cal_pd = (
  lifetimes.utils.calibration_and_holdout_data(
    orders_pd,
    customer_id_col='CustomerID',
    datetime_col='InvoiceDate',
    observation_period_end = current_date,
    calibration_period_end=calibration_end_date,
    freq='D'    
    )
  )

# display first few rows
metrics_cal_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC 前回同様、Spark SQLを利用して同じ情報を得ることができます。ここでも、SQL文とSQL API(Pythonなどから呼び出す)の両方を使って検討します。
# MAGIC 
# MAGIC このSQL文を理解するには、まず2つの主要部分に分かれていることを認識してください。
# MAGIC 
# MAGIC 1つ目の部分では、前のクエリの例で行ったのと同様に、校正期間の顧客ごとにコアメトリクス、すなわち再帰性、頻度、年齢（T）を計算します。
# MAGIC 
# MAGIC 2つ目の部分では、顧客ごとにホールドアウトした顧客の購入日の数を計算します。この値(frequency_holdout)は、キャリブレーション期間とホールドアウト期間の両方にわたる顧客の全取引履歴を調べたときに、校正期間の頻度(frequency_cal)に追加される増分を表しています。
# MAGIC 
# MAGIC 
# MAGIC ロジックを単純化するために、CustomerHistoryという名前の共通テーブル式（CTE）をクエリの先頭に定義しています。このクエリは、顧客の取引履歴を構成する関連する日付を抽出するもので、前回調べたSQL文の中心にあるロジックをよく反映しています。唯一の違いは、保留期間の日数(duration_holdout)を含めていることです。

# COMMAND ----------

# DBTITLE 1,SQL(Spark SQL)を使って算出
sql = '''
WITH CustomerHistory 
  AS (
    SELECT  -- nesting req'ed b/c can't SELECT DISTINCT on widget parameter
      m.*,
      getArgument('holdout days') as duration_holdout
    FROM (
      SELECT DISTINCT
        x.customerid,
        z.first_at,
        TO_DATE(x.invoicedate) as transaction_at,
        y.current_dt
      FROM orders x
      CROSS JOIN (SELECT MAX(TO_DATE(invoicedate)) as current_dt FROM orders) y                                -- current date (according to dataset)
      INNER JOIN (SELECT customerid, MIN(TO_DATE(invoicedate)) as first_at FROM orders GROUP BY customerid) z  -- first order per customer
        ON x.customerid=z.customerid
      WHERE x.customerid IS NOT NULL
    ) m
  )
SELECT
    a.customerid as CustomerID,
    a.frequency as frequency_cal,
    a.recency as recency_cal,
    a.T as T_cal,
    COALESCE(b.frequency_holdout, 0.0) as frequency_holdout,
    a.duration_holdout
FROM ( -- CALIBRATION PERIOD CALCULATIONS
    SELECT
        p.customerid,
        CAST(p.duration_holdout as float) as duration_holdout,
        CAST(DATEDIFF(MAX(p.transaction_at), p.first_at) as float) as recency,
        CAST(COUNT(DISTINCT p.transaction_at) - 1 as float) as frequency,
        CAST(DATEDIFF(DATE_SUB(p.current_dt, int(p.duration_holdout)), p.first_at) as float) as T
    FROM CustomerHistory p
    WHERE p.transaction_at < DATE_SUB(p.current_dt, int(p.duration_holdout))  -- LIMIT THIS QUERY TO DATA IN THE CALIBRATION PERIOD
    GROUP BY p.customerid, p.first_at, p.current_dt, p.duration_holdout
  ) a
LEFT OUTER JOIN ( -- HOLDOUT PERIOD CALCULATIONS
  SELECT
    p.customerid,
    CAST(COUNT(DISTINCT p.transaction_at) as float) as frequency_holdout
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

# MAGIC %md
# MAGIC 
# MAGIC そして、これに相当するProgrammatic SQL APIのロジックは以下の通りです。

# COMMAND ----------

# DBTITLE 1,Python(pyspark)を使って算出
from pyspark.sql.functions import avg, date_sub, coalesce, lit, expr

# valid customer orders
x = orders.where(orders.CustomerID.isNotNull())

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
      to_date(x.InvoiceDate).alias('transaction_at'), 
      y.current_dt, 
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
         datediff( expr('date_sub(current_dt, duration_holdout)'), p.first_at).cast(FloatType()).alias('T_cal')
       )
    )

b = (p
      .where((p.transaction_at >= expr('date_sub(current_dt, duration_holdout)')) & (p.transaction_at <= p.current_dt) )
      .groupBy(p.customerid)
      .agg(
        countDistinct(p.transaction_at).cast(FloatType()).alias('frequency_holdout')
        )
   )

metrics_cal_api = (a
                 .join(b, a.customerid==b.customerid, how='left')
                 .select(
                   a.customerid.alias('CustomerID'),
                   a.frequency_cal,
                   a.recency_cal,
                   a.T_cal,
                   coalesce(b.frequency_holdout, lit(0.0)).alias('frequency_holdout'),
                   a.duration_holdout
                   )
                 .orderBy('CustomerID')
              )

display(metrics_cal_api)

# COMMAND ----------

# MAGIC %md 
# MAGIC サマリー統計を使って、これらの異なるロジックのユニットが同じ結果を返していることを再度確認することができます。

# COMMAND ----------

# DBTITLE 1,lifetimesライブラリを使って算出 - 結果確認
# summary data from lifetimes
metrics_cal_pd.describe()

# COMMAND ----------

# DBTITLE 1,SQL(Spark SQL)を使って算出 - 結果確認
# summary data from SQL statement
metrics_cal_sql.toPandas().describe()

# COMMAND ----------

# DBTITLE 1,Python(pyspark)を使って算出 - 結果確認
# summary data from pyspark.sql API
metrics_cal_api.toPandas().describe()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC データの準備はほぼ完了しています。最後に、リピート購入のない顧客を除外します。すなわち、frequencyまたはfrequency_calが0の場合です。
# MAGIC 
# MAGIC 使用するPareto/NBDおよびBG/NBDモデルは、リピート取引のある顧客に対してのみ計算を行うことに焦点を当てています。BG/NBDモデルを修正したMBG/NBDモデルは、リピート購入のない顧客を対象としており、lifetimesライブラリでサポートされています。しかし、現在使用されているBYTDモデルの中で最も人気のある2つのモデルにこだわるため、データをその要件に合わせて制限します。
# MAGIC 
# MAGIC 
# MAGIC 注: ノートブックのこのセクションで以前に行ったサイド・バイ・サイドの比較との一貫性を保つために、pandasとSpark DataFramesの両方にフィルターをかける方法を示しています。 実際の実装では、データの準備にpandasとSpark DataFramesのどちらを使用するかを選択するだけです。

# COMMAND ----------

# リピートなしの顧客を除外(フルデータセットの方のDataFrame)
filtered_pd = metrics_pd[metrics_pd['frequency'] > 0]
filtered = metrics_api.where(metrics_api.frequency > 0)

## リピートなしの顧客を除外(キャリブレーションのDataFrame)
filtered_cal_pd = metrics_cal_pd[metrics_cal_pd['frequency_cal'] > 0]
filtered_cal = metrics_cal_api.where(metrics_cal_api.frequency_cal > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 4: モデルのトレーニング(学習)
# MAGIC 
# MAGIC 
# MAGIC モデルのトレーニングを簡単に行うために、[パレート/NBDモデル（オリジナルのBTYDモデル）](https://www.jstor.org/stable/2631608)を使用した簡単な演習から始めましょう。このノートブックの最後のセクションで構築されたキャリブレーション-ホールドアウトデータセットを使用して、キャリブレーションデータにモデルをフィットさせ、後でホールドアウトデータを使用して評価します。

# COMMAND ----------

from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.fitters.beta_geo_fitter import BetaGeoFitter

# Spark DFをPandas DFに変換
input_pd = filtered_cal.toPandas()

# モデルの学習
model = ParetoNBDFitter(penalizer_coef=0.0)
model.fit( input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC モデルがフィットしたところで、ホールドアウト期間の予測をしてみましょう。次のステップで比較できるように、同じ期間の実績値を取得します。

# COMMAND ----------

# ホールドアウト期間のFrequencyを学習したモデルを使って推測する
frequency_holdout_predicted = model.predict( input_pd['duration_holdout'], input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# 実際のFrequency
frequency_holdout_actual = input_pd['frequency_holdout']

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 実際の値と予測値があれば、いくつかの標準的な評価指標を計算することができます。 今後の評価を容易にするために、これらの計算を関数にまとめておきましょう。

# COMMAND ----------

import numpy as np

def score_model(actuals, predicted, metric='mse'):
  # metric名を小文字に統一して扱う
  metric = metric.lower()
  
  # metric引数が、二乗誤差平均(mse)と標準偏差(rmse)の場合
  if metric=='mse' or metric=='rmse':
    val = np.sum(np.square(actuals-predicted))/actuals.shape[0]
    if metric=='rmse':
        val = np.sqrt(val)
  
  # metric引数が、平均絶対誤差(mae)の場合
  elif metric=='mae':
    np.sum(np.abs(actuals-predicted))/actuals.shape[0]
  
  else:
    val = None
  
  return val

# 上記で定義した関数`score_model()`を使って、モデルの二乗誤差平均(mse)を計算・表示
print('MSE: {0}'.format(score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC パレート/NBDモデルの内部は非常に複雑であるかもしれません。簡単に言うと、このモデルは2つの曲線の二重積分を計算します。1つは母集団内の顧客の購入頻度を表し、もう1つは以前の購入イベント後の顧客の生存率を表します。すべての計算ロジックは、lifetimesライブラリによってシンプルなメソッドコールになっています。
# MAGIC 
# MAGIC モデルをトレーニングするのは簡単かもしれませんが、ここでは、パレート/NBDモデルとBG/NBDモデルの2つのモデルを使用することができます。
# MAGIC [BG/NBDモデル](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)は、顧客寿命の計算を単純化したモデルで、BTYDアプローチを広めたモデルでもあります。どちらのモデルも、同じ顧客の特徴をもとに、同じ制約条件を採用しています。(2つのモデルの主な違いは、BG/NBDモデルでは、生存曲線をパレート分布ではなくベータ幾何分布にマッピングすることです)。可能な限り最高の適合性を得るために、両モデルの結果を我々のデータセットと比較することは価値のあることです。
# MAGIC 
# MAGIC 
# MAGIC それぞれのモデルは，L2ノルム正則化パラメータを利用します． どのモデルが最も優れているかを調べるだけでなく，このパラメータにどの値（0と1の間）が最も適しているかを検討する必要があります． これにより、ハイパーパラメータを調整することで、かなり広い探索空間を得ることができます。
# MAGIC 
# MAGIC この作業を支援するために、[Hyperopt](http://hyperopt.github.io/hyperopt/)を利用します。Hyperoptは、ハイパーパラメータ探索空間に対するモデルのトレーニングと評価を並列化します。これは、1台のマシンのマルチプロセッサ・リソースを活用することも、Sparkクラスタが提供する広範なリソースを活用することもできます。モデルを反復するたびに、損失関数が計算されます。様々な最適化アルゴリズムを用いて、hyperoptは探索空間をナビゲートし、損失関数によって返される値を最小化するために利用可能なパラメータ設定の最適な組み合わせを見つけます。
# MAGIC 
# MAGIC 
# MAGIC Hyperoptを利用するには、探索空間を定義し、モデルの学習と評価のロジックを書き換えて、損失関数の測定値を返す単一の関数呼び出しを提供します。

# COMMAND ----------

from hyperopt import hp, fmin, tpe, rand, SparkTrials, STATUS_OK, STATUS_FAIL, space_eval

# サーチする範囲を指定する
search_space = hp.choice('model_type',[
                  {'type':'Pareto/NBD', 'l2':hp.uniform('pareto_nbd_l2', 0.0, 1.0)},
                  {'type':'BG/NBD'    , 'l2':hp.uniform('bg_nbd_l2', 0.0, 1.0)}  
                  ]
                )

# モデルを評価する関数を定義する
def evaluate_model(params):
  
  # accesss replicated input_pd dataframe
  data = inputs.value
  
  # 外部から与えられるパラメータをキャッチする
  model_type = params['type']
  l2_reg = params['l2']
  
  # モデルの初期化
  if model_type == 'BG/NBD':
    model = BetaGeoFitter(penalizer_coef=l2_reg)
  elif model_type == 'Pareto/NBD':
    model = ParetoNBDFitter(penalizer_coef=l2_reg)
  else:
    return {'loss': None, 'status': STATUS_FAIL}
  
  # モデルの学習
  model.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
  
  # モデルの評価
  frequency_holdout_actual = data['frequency_holdout']
  frequency_holdout_predicted = model.predict(data['duration_holdout'], data['frequency_cal'], data['recency_cal'], data['T_cal'])
  mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')
  
  # スコアとステータスを返す
  return {'loss': mse, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC evaluate_model関数は、inputsという変数からデータを取得していることに注目してください。inputsは次のセルで、先ほど使用したinputs_pd DataFrameを含む[ブロードキャスト変数](https://spark.apache.org/docs/latest/rdd-programming-guide.html#broadcast-variables)として定義されています。ブロードキャスト変数として、モデルが使用するデータセットの完全なスタンドアローンコピーがSparkクラスタの各ワーカーに複製されます。これにより、hyperoptの反復ごとにクラスタ・ドライバからワーカーに送信しなければならないデータ量が制限されます。このドキュメントやその他のhyperoptのベストプラクティスについては、[こちら](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/hyperopt-best-practices.html)を参照してください。
# MAGIC 
# MAGIC すべての準備が整ったところで、データセットに最適なモデルタイプとL2設定を特定するために、100回の反復でハイパーパラメータのチューニングを実行してみましょう。

# COMMAND ----------

import mlflow

# input_pd(Pandas DF)をsparkクラスタにコピーして配っておく
inputs = sc.broadcast(input_pd)

# configure hyperopt settings to distribute to all executors on workers
spark_trials = SparkTrials(parallelism=2)

# select optimization algorithm
algo = tpe.suggest

# perform hyperparameter tuning (logging iterations to mlflow)
argmin = fmin(
  fn=evaluate_model,
  space=search_space,
  algo=algo,
  max_evals=100,
  trials=spark_trials
  )

# release the broadcast dataset
inputs.unpersist()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC DatabricksのMLランタイムと一緒に使用すると、探索空間の評価を構成する個々のランがmlflowと呼ばれる組み込みのリポジトリで追跡されます。Databricksのmlflowインターフェースを使ってhyperoptが生成したモデルをレビューする方法については、[こちらのドキュメント](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/hyperopt-spark-mlflow-integration.html)をご覧ください。
# MAGIC 
# MAGIC hyperoptの反復の間に観察された最適なハイパーパラメータの設定は、argmin変数に取り込まれます。 space_eval関数を使用することで，どの設定が最も優れているかをわかりやすく表現することができます．

# COMMAND ----------

# print optimum hyperparameter settings
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md 
# MAGIC さて、*最適な* パラメータ設定がわかったところで、このパラメータを使ってモデルをトレーニングし、より詳細なモデル評価を行ってみましょう。
# MAGIC 
# MAGIC 注：検索スペースの検索方法が異なるため、hyperoptの実行結果が若干異なる場合があります。

# COMMAND ----------

# get hyperparameter settings
params = space_eval(search_space, argmin)
model_type = params['type']
l2_reg = params['l2']

# instantiate and configure model
if model_type == 'BG/NBD':
  model = BetaGeoFitter(penalizer_coef=l2_reg)
elif model_type == 'Pareto/NBD':
  model = ParetoNBDFitter(penalizer_coef=l2_reg)
else:
  raise 'Unrecognized model type'
  
# train the model
model.fit(input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ###Step 5: モデルの評価
# MAGIC 
# MAGIC このノートの最後のセクションで定義された方法を使って、新しく学習されたモデルのMSEを計算することができます。

# COMMAND ----------

# score the model
frequency_holdout_actual = input_pd['frequency_holdout']
frequency_holdout_predicted = model.predict(input_pd['duration_holdout'], input_pd['frequency_cal'], input_pd['recency_cal'], input_pd['T_cal'])
mse = score_model(frequency_holdout_actual, frequency_holdout_predicted, 'mse')

print('MSE: {0}'.format(mse))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC モデルを比較するには重要ですが、個々のモデルの全体的な適合性という点では、MSE指標の解釈は少し難しいです。モデルがデータにどれだけフィットしているかをより深く理解するために、いくつかの実測値と予測値の関係を視覚化してみましょう。
# MAGIC 
# MAGIC まず、キャリブレーション期間中の購入頻度が、ホールドアウト期間中の実際の頻度（frequency_holdout）および予測頻度（model_predictions）とどのように関連しているかを調べてみます。

# COMMAND ----------

from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  n=90, 
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ここでわかることは、キャリブレーション期間の購入回数が多いほど、ホールドアウト期間の平均購入回数も多くなることが予測されます。ただし、キャリブレーション期間の購入回数が多い（60回以上）お客様を考慮すると、実際の値はモデルの予測から大きく乖離します。
# MAGIC 
# MAGIC このNotebookの「データ探索」ステップで可視化したチャートによると、このように多くの購入回数を持つ顧客はほとんどいないので、この乖離は頻度の高い範囲のごく限られた事例の結果である可能性があります。より多くのデータがあれば、予測値と実測値がこの高域で再び一致するかもしれません。この乖離が続くようであれば、信頼できる予測ができない顧客エンゲージメント頻度の範囲を示しているのかもしれません。
# MAGIC 
# MAGIC 同じメソッドコールを使用して、ホールドアウト期間中の平均購入回数に対する最終購入からの時間を視覚化することができます。この図では、最終購入からの時間が長くなるにつれて、ホールドアウト期間中の購入数が減少していることがわかります。 つまり、しばらく会っていないお客様は、すぐには戻ってこない可能性が高いということです。
# MAGIC 
# MAGIC 
# MAGIC 注: 前回同様、ビジュアライゼーションに集中するため、以下のセルではコードを隠します。関連するPythonロジックを見るには **"Show code"** を使用してください。

# COMMAND ----------

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  kind='time_since_last_purchase', 
  n=90, 
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC このグラフに最終購入時の顧客の経過日数(Age)を入れてみると、顧客ライフサイクルにおける最終購入のタイミングは、顧客の経過日数が大きくなるまで、ホールドアウト期間の購入回数にはあまり影響しないようです。これは、長く付き合ってくれる顧客は、より頻繁に購入してくれる可能性が高いことを示しています。

# COMMAND ----------

plot_calibration_purchases_vs_holdout_purchases(
  model, 
  input_pd, 
  kind='recency_cal', 
  n=300,
  **{'figsize':(8,8)}
  )

display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ざっと見たところ、このモデルは完璧ではありませんが、いくつかの有用なパターンを捉えていることがわかります。これらのパターンを使って、顧客がエンゲージメントを維持する確率を計算することができます。

# COMMAND ----------

# add a field with the probability a customer is currently "alive"
filtered_pd['prob_alive']=model.conditional_probability_alive(
    filtered_pd['frequency'], 
    filtered_pd['recency'], 
    filtered_pd['T']
    )

filtered_pd.head(10)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 顧客が生存している確率の予測は、モデルをマーケティングプロセスに応用する上で非常に興味深いものとなるでしょう。しかし、モデルの展開を検討する前に、データセットの中で活動が控えめな1人の顧客（CustomerID 12383）の履歴を見て、この確率が顧客の再活動に伴ってどのように変化するかを見てみましょう。

# COMMAND ----------

from lifetimes.plotting import plot_history_alive
import matplotlib.pyplot as plt

# clear past visualization instructions
plt.clf()

# customer of interest
CustomerID = '12383'

# grab customer's metrics and transaction history
cmetrics_pd = input_pd[input_pd['CustomerID']==CustomerID]
trans_history = orders_pd.loc[orders_pd['CustomerID'] == CustomerID]

# calculate age at end of dataset
days_since_birth = 400

# plot history of being "alive"
plot_history_alive(
  model, 
  days_since_birth, 
  trans_history, 
  'InvoiceDate'
  )

display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC このグラフから、この顧客が2011年1月に最初の購入を行い、その月の後半にリピート購入を行ったことがわかります。その後、約1ヶ月間の活動休止期間があり、この顧客が生存している確率はわずかに低下しましたが、同年3月、4月、6月に購入された際には、顧客が活動していることを示すシグナルが繰り返し発信されました。最後の6月の購入以降、その顧客は取引履歴に現れず、顧客が生きているという確信は、それまでのシグナルを考えると緩やかなペースで低下しています。
# MAGIC 
# MAGIC モデルはどのようにしてこれらの確率を算出しているのでしょうか？正確な計算は難しいのですが、生存している確率を頻度と反復性に関連づけてヒートマップとしてプロットすることで、これら2つの値の交点に割り当てられた確率を理解することができます。

# COMMAND ----------

from lifetimes.plotting import plot_probability_alive_matrix

# set figure size
plt.subplots(figsize=(12, 8))

plot_probability_alive_matrix(model)

display()

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 顧客の生存確率を予測するだけでなく、今後30日間のような将来のある時間間隔において、予想される顧客の購入数を算出することができます。

# COMMAND ----------

from lifetimes.plotting import plot_frequency_recency_matrix

# set figure size
plt.subplots(figsize=(12, 8))

plot_frequency_recency_matrix(model, T=30)

display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 先ほどと同様に、現在の指標に基づいて、お客様ごとにこの確率を計算することができます。

# COMMAND ----------

filtered_pd['purchases_next30days']=(
  model.conditional_expected_number_of_purchases_up_to_time(
    30, 
    filtered_pd['frequency'], 
    filtered_pd['recency'], 
    filtered_pd['T']
    )
  )

filtered_pd.head(10)

# COMMAND ----------

# MAGIC %md ###Step 6: モデルの展開と予測計算(スコアリング)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 学習されたBTYDモデルを使用する方法は数多くあります。 顧客がまだ契約を継続している可能性を理解することができます。 また、ある日数の間に顧客が購入すると予想される数を予測することもできます。 これらの予測を行うために必要なのは、トレーニングされたモデルと、頻度(frequency)、反復性(recency)、および顧客の経過日数(Age, T)の値だけです。

# COMMAND ----------

frequency = 6
recency = 255
T = 300
t = 30

print('Probability of Alive: {0}'.format( model.conditional_probability_alive(frequency, recency, T) ))
print('Expected Purchases in next {0} days: {1}'.format(t, model.conditional_expected_number_of_purchases_up_to_time(t, frequency, recency, T) ))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 
# MAGIC ここでの課題は、モデルをパッケージ化して、この目的に再利用できるようにすることです。前述したように、ハイパーパラメータ・チューニングの際には、mlflowとhyperoptを組み合わせて、モデルの実行をキャプチャしました。プラットフォームとしてのmlflowは、関数としてのモデルやマイクロサービス・アプリケーションのデプロイメントなど、モデルの開発やデプロイメントに伴うさまざまな課題を解決するために設計されています。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC MLflowは、[多くの一般的なモデルタイプ](https://www.mlflow.org/docs/latest/models.html#built-in-model-flavors)に対して、すぐに導入の課題に取り組むことができます。しかし、ライフタイム・モデルはその一つではありません。mlflowをデプロイメントビークルとして使用するためには、標準的なmlflowのAPIコールをモデルに適用できるロジックに変換するカスタムラッパークラスを書く必要があります。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これを説明するために、寿命モデルのラッパークラスを実装しました。このラッパークラスは、モデルに対する複数の予測呼び出しにmlflowのpredict()メソッドをマッピングします。通常、predict()を単一の予測にマッピングしますが、カスタムロジックを実装するためにラッパーを使用する多くの方法の1つを示すために、返される結果の複雑さを増加させました。

# COMMAND ----------

import mlflow
import mlflow.pyfunc

# create wrapper for lifetimes model
class _lifetimesModelWrapper(mlflow.pyfunc.PythonModel):
  
    def __init__(self, lifetimes_model):
        self.lifetimes_model = lifetimes_model

    def predict(self, context, dataframe):
      
      # access input series
      frequency = dataframe.iloc[:,0]
      recency = dataframe.iloc[:,1]
      T = dataframe.iloc[:,2]
      
      # calculate probability currently alive
      results = pd.DataFrame( 
                  self.lifetimes_model.conditional_probability_alive(frequency, recency, T),
                  columns=['alive']
                  )
      # calculate expected purchases for provided time period
      results['purch_15day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(15, frequency, recency, T)
      results['purch_30day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(30, frequency, recency, T)
      results['purch_45day'] = self.lifetimes_model.conditional_expected_number_of_purchases_up_to_time(45, frequency, recency, T)
      
      return results[['alive', 'purch_15day', 'purch_30day', 'purch_45day']]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 次に、モデルを mlflow に登録する必要があります。登録の際には、モデルの機能に期待されるAPIをマッピングするラッパーを通知します。また、モデルを動作させるために、どのライブラリをインストールしてロードする必要があるかを指示するために、実行環境の情報を提供します。
# MAGIC 
# MAGIC 
# MAGIC 通常、モデルのトレーニングとロギングは1つのステップとして行われますが、このノートブックでは、カスタムモデルのデプロイメントに焦点を当てるために、2つのアクションを分離しました。より一般的なmlflowの実装パターンについては、[こちら](https://docs.databricks.com/applications/mlflow/model-example.html)やオンラインで公開されている他の例を参照してください。

# COMMAND ----------

# add lifetimes to conda environment info
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][-1]['pip'] += ['lifetimes==0.10.1'] # version should match version installed at top of this notebook

# save model run to mlflow
with mlflow.start_run(run_name='deployment run') as run:
  mlflow.pyfunc.log_model(
    'model', 
    python_model=_lifetimesModelWrapper(model), 
    conda_env=conda_env
    )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC モデルに依存関係の情報とクラスのラッパーが記録されたので、mlflowを使ってモデルをSpark DataFrameに対して使用できる関数に変換してみましょう。

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType

# define the schema of the values returned by the function
result_schema = ArrayType(FloatType())

# define function based on mlflow recorded model
probability_alive_udf = mlflow.pyfunc.spark_udf(
  spark, 
  'runs:/{0}/model'.format(run.info.run_id), 
  result_type=result_schema
  )

# register the function for use in SQL
_ = spark.udf.register('probability_alive', probability_alive_udf)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 顧客のFrequency, recencyおよびageを与えれば、上記で定義したUDFで推定(スコアリング)が可能になります。

# COMMAND ----------

# create a temp view for SQL demonstration (next cell)
filtered.createOrReplaceTempView('customer_metrics')

# demonstrate function call on Spark DataFrame
display(
  filtered
    .withColumn(
      'predictions', 
      probability_alive_udf(filtered.frequency, filtered.recency, filtered.T)
      )
    .selectExpr(
      'customerid', 
      'predictions[0] as prob_alive', 
      'predictions[1] as purch_15day', 
      'predictions[2] as purch_30day', 
      'predictions[3] as purch_45day'
      )
  )

# COMMAND ----------

# MAGIC %sql -- predict probabiliies customer is alive and will return in 15, 30 & 45 days
# MAGIC 
# MAGIC SELECT
# MAGIC   x.CustomerID,
# MAGIC   x.prediction[0] as prob_alive,
# MAGIC   x.prediction[1] as purch_15day,
# MAGIC   x.prediction[2] as purch_30day,
# MAGIC   x.prediction[3] as purch_45day
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CustomerID,
# MAGIC     probability_alive(frequency, recency, T) as prediction
# MAGIC   FROM customer_metrics
# MAGIC   ) x;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ライフタイムモデルが関数として登録されたことで、ETLバッチ、ストリーミング、インタラクティブクエリのワークロードに顧客の生涯確率を組み込むことができるようになりました。また、mlflowのモデル展開機能を利用して、[AzureML](https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-quick-start-deployment-azure.html)や[AWS Sagemaker](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-deployment-aws.html)を利用したスタンドアロンのWebサービスとしても展開することができます。
