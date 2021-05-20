# Databricks notebook source
# MAGIC %md
# MAGIC ## サンプルデータの準備

# COMMAND ----------

# DBTITLE 1,サンプルのcsvファイルをダウンロードしてsparkから参照可能なストレージ上に配置
# MAGIC %sh
# MAGIC wget https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/sample-data/1.csv && \
# MAGIC wget https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/sample-data/2.csv && \
# MAGIC wget https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/sample-data/3.csv
# MAGIC 
# MAGIC mkdir -p /dbfs/FileStore/tmp/sample-data/
# MAGIC mv {1,2,3}.csv /dbfs/FileStore/tmp/sample-data/
# MAGIC 
# MAGIC ls -l /dbfs/FileStore/tmp/sample-data/

# COMMAND ----------

# DBTITLE 1,CSVファイルの中身を確認(3ファイルで日毎にstock数だけ変化するデータ)
# MAGIC %fs  head dbfs:/FileStore/tmp/sample-data/1.csv

# COMMAND ----------

# MAGIC %fs  head dbfs:/FileStore/tmp/sample-data/2.csv

# COMMAND ----------

# MAGIC %fs  head dbfs:/FileStore/tmp/sample-data/3.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 目的: ４種類のプロダクトの在庫数(stock)の変化(前日との差分)を見る
# MAGIC 
# MAGIC * データ: 日次在庫レポート:　プロダクト名(productid)と在庫数(stock)のみ含む 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 方式1: Delta Lakeのタイムトラベル機能を使う
# MAGIC 
# MAGIC * 利点: 明示的にタイムスタンプをカラムに追加しなくても良い
# MAGIC * 欠点: 日毎にSQLクエリを書く必要がある。例えば、30日間の前日比データを見るには、30回個別にクエリを書かないといけない。
# MAGIC 
# MAGIC 上記の通り、過去の2,3程度の時点のみを比較するには向いているが、時系列で全ての期間のデータを作るのには向いていない。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 一つ目のファイルを読み込んで、Deltaテーブルを作る

# COMMAND ----------

from pyspark.sql.functions import *

schema = '''
  productid string,
  stock int
'''

# csvファイルの読み込み
df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/1.csv')
)

# 読み込んだdataframeの確認
display(df)

# COMMAND ----------

# deltaへの書き込み(ファイル)
df.write.format('delta').mode('append').save('dbfs:/FileStore/tmp/delta/product_stock')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- deltaテーブルの作成(メタストア)
# MAGIC create table if not exists product_stock
# MAGIC using delta
# MAGIC location "dbfs:/FileStore/tmp/delta/product_stock"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- deltaテーブルの確認
# MAGIC select * from product_stock

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2つめのcsvを読み込み、既存のDeltaテーブルにMergeする(追記する)

# COMMAND ----------

schema = '''
  productid string,
  stock int
'''

df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/2.csv')
)

df.createOrReplaceTempView('tmp_product_stock')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 上記で読み込んだデータの確認(tmep view)
# MAGIC select * from tmp_product_stock

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- 既存のDeltaテーブルに追記(Merge)
# MAGIC -- 　　つまり、元テーブル(product_stock)に対して、
# MAGIC --    今読み込んだテーブル(tmp_product_stock)をMergeする
# MAGIC merge into product_stock
# MAGIC using tmp_product_stock
# MAGIC on product_stock.productid = tmp_product_stock.productid
# MAGIC when matched then
# MAGIC   update set *
# MAGIC when not matched then
# MAGIC   insert *

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Merge後のテーブルを参照
# MAGIC select * from product_stock

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Deltaのスナップショットhistoryを確認する
# MAGIC describe history product_stock

# COMMAND ----------

# MAGIC %md
# MAGIC #### 同様に、3つめのファイルを読み込み、既存のDeltaテーブルにMergeする(追記する)

# COMMAND ----------

# 2つ目のファイルの処理と同等なので、本来は関数にすべき。
# ここでは簡単のため、同じコードを2回かく

schema = '''
  productid string,
  stock int
'''

df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/3.csv')  # <==ファイル名だけ変わっている
)

df.createOrReplaceTempView('tmp_product_stock')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from tmp_product_stock

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Merge
# MAGIC merge into product_stock
# MAGIC using tmp_product_stock
# MAGIC on product_stock.productid = tmp_product_stock.productid
# MAGIC when matched then
# MAGIC   update set *
# MAGIC when not matched then
# MAGIC   insert *

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Merge後のテーブルを参照
# MAGIC select * from product_stock

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Deltaのスナップショットhistoryを確認する
# MAGIC -- Write -> Merge -> Mergeの順に履歴が刻まれている!
# MAGIC describe history product_stock

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### TimeTravelの機能を使って、ver.2とver.1のstock数の比較をする

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Delta Lakeのタイムトラベル機能:
# MAGIC -- "version as of xxx"で過去のversionを指定
# MAGIC -- これを"timestamp as of <タイムスタンプ>"で指定することで、日時指定も可能
# MAGIC select
# MAGIC   cur.productid,
# MAGIC   cur.stock as cur_stock,
# MAGIC   prev.stock as prev_stock,
# MAGIC   (cur.stock - prev.stock) as diff_stock
# MAGIC from
# MAGIC   prodstock cur -- <==　現在のテーブル
# MAGIC   left join (
# MAGIC     select
# MAGIC       *
# MAGIC     from
# MAGIC       prodstock version as of 1 -- <== 過去のテーブル
# MAGIC   ) prev on cur.productid = prev.productid

# COMMAND ----------

# MAGIC %md
# MAGIC 注意: 結局、上ではver2とver1の比較しかしていない。ver1とver0の比較は、同様なことをまた別のクエリとして書く必要がある。

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 方式2: Timestamp付与方式(主にPython版)
# MAGIC 
# MAGIC データをDeltaに取り込むときにタイムスタンプを(マニュアルで)付与する方式。
# MAGIC Deltaにするときに組み込み関数 `current_timestapm()`でカラムを一つ追加するだけなので、そこまで稼働は増えない。

# COMMAND ----------

# MAGIC %md
# MAGIC #### 一つ目のファイルを読み込んで、Deltaテーブルを作る。その際、timestampも付与する。

# COMMAND ----------

schema = '''
  productid string,
  stock int
'''

# CSVファイルを読み込み
df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/1.csv')
)

# Timestampのカラム(ts)を追加
df_with_ts = (
  df
  .select("*")
  .withColumn('ts', current_date()) # <== timestampのカラムを追加 (by Python)
)

# タイムスタンプつきのDataFrameを確認する
display(df_with_ts)

# Deltaに書き出し、Deltaテーブルも作成する
df_with_ts.write.format('delta').mode('append').save('dbfs:/FileStore/tmp/delta/product_stock_ts')

spark.sql('''
    create table if not exists prodstock_ts 
    using delta 
    location "dbfs:/FileStore/tmp/delta/product_stock_ts"
''')

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2つめのcsvを読み込み、既存のDeltaテーブルにMergeする(追記する)

# COMMAND ----------

# csvファイルの読み込み
df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/2.csv')
)

# temp viewの作成
df.createOrReplaceTempView('tmp_product_stock')

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Merge
# MAGIC merge into prodstock_ts
# MAGIC using 
# MAGIC (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     date_add( current_date(), 1) as ts -- <== Timestamp付与処理。本来は `current_date() as ts` でよいが、デモのため日付をずらすため、date_add()で1日ずらしている
# MAGIC   FROM tmp_product_stock
# MAGIC ) AS tmp_product_stock_ts
# MAGIC on prodstock_ts.productid = tmp_product_stock_ts.productid
# MAGIC   and prodstock_ts.ts = tmp_product_stock_ts.ts
# MAGIC when matched then
# MAGIC   update set *
# MAGIC when not matched then
# MAGIC   insert *

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prodstock_ts

# COMMAND ----------

# MAGIC %md
# MAGIC #### 同様に、3つめのcsvを読み込み、既存のDeltaテーブルにMergeする(追記する)

# COMMAND ----------

# csvファイルの読み込み
df = (
  spark.read
  .format('csv')
  .schema(schema)
  .option('Header', False)
  .load('dbfs:/FileStore/tmp/sample-data/3.csv')
)

# temp viewの作成
df.createOrReplaceTempView('tmp_product_stock')

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Merge
# MAGIC merge into prodstock_ts
# MAGIC using 
# MAGIC (
# MAGIC   SELECT
# MAGIC     *,
# MAGIC     date_add( current_date(), 2) as ts -- <== Timestamp付与処理。本来は `current_date() as ts` でよいが、デモのため日付をずらすため、date_add()で2日ずらしている
# MAGIC   FROM tmp_product_stock
# MAGIC ) AS tmp_product_stock_ts
# MAGIC on prodstock_ts.productid = tmp_product_stock_ts.productid
# MAGIC   and prodstock_ts.ts = tmp_product_stock_ts.ts
# MAGIC when matched then
# MAGIC   update set *
# MAGIC when not matched then
# MAGIC   insert *

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from prodstock_ts

# COMMAND ----------

# MAGIC %md
# MAGIC #### SQL/joinを使って、一日前のstock数をカラムに追加。加えて、そこから差分もカラム化

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   cur.productid,
# MAGIC   cur.stock,
# MAGIC   cur.ts,
# MAGIC   prev.stock,
# MAGIC   prev.ts,
# MAGIC   (cur.stock - prev.stock) as diff_stock
# MAGIC from
# MAGIC   prodstock_ts cur
# MAGIC   left join (
# MAGIC     select
# MAGIC       *
# MAGIC     from
# MAGIC       prodstock_ts --where ts = date_sub( cur.ts, 1)
# MAGIC   ) as prev on cur.productid = prev.productid
# MAGIC   and date_sub(cur.ts, 1) = prev.ts
# MAGIC order by cur.ts;

# COMMAND ----------

# MAGIC %md ### 環境のクリーンアップ

# COMMAND ----------

# DBTITLE 1,環境のクリーンアップ1 (テーブル削除)
# MAGIC %sql
# MAGIC --レーブルの削除
# MAGIC DROP TABLE prodstock;
# MAGIC DROP TABLE prodstock_ts

# COMMAND ----------

# DBTITLE 1,環境のクリーンアップ2 (ファイル削除)
# MAGIC %fs rm -r dbfs:/FileStore/tmp/delta/product_stock

# COMMAND ----------

dbutils.fs.rm('dbfs:/FileStore/tmp/delta/product_stock', recurse=True)

# COMMAND ----------

# MAGIC %fs rm -r dbfs:/FileStore/tmp/delta/product_stock_ts 
