-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Databricksで実践するData &amp; AI
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC 本デモでは、Databricksの基本的な使い方を、EDA(Exploratory Data Analysis)の流れに沿ってご説明します。また、Databricks特有のDelta Lake、ストリーミング処理をご紹介します。
-- MAGIC <table>
-- MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
-- MAGIC   <tr><td>日付</td><td>2021/01/18</td></tr>
-- MAGIC   <tr><td>バージョン</td><td>1.1</td></tr>
-- MAGIC   <tr><td>クラスター</td><td>7.4ML</td></tr>
-- MAGIC </table>
-- MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 基本的な使い方
-- MAGIC ### クラスターを作成
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
-- MAGIC (1) 左のサイドバーの **Clusters** を右クリック。新しいタブもしくはウィンドウを開く。<br>
-- MAGIC (2) クラスタページにおいて **Create Cluster** をクリック。<br>
-- MAGIC (3) クラスター名を **Quickstart** とする。<br>
-- MAGIC (4) Databricks ランタイム バージョン を ドロップダウンし、 例えば **7.4ML (Scala 2.12, Spark 3.0.0)** を選択。<br>
-- MAGIC (5) 最後に **Create Cluster** をクリックすると、クラスターが起動 !
-- MAGIC </div>
-- MAGIC 
-- MAGIC ### ノートブックを作成したクラスターに紐付けて、 run all コマンドを実行
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
-- MAGIC (1) ノートブックに戻ります。<br>
-- MAGIC (2) 左上の ノートブック メニューバーから、 **<img src="http://docs.databricks.com/_static/images/notebooks/detached.png"/></a> > Quickstart** を選択。<br>
-- MAGIC (3) クラスターが <img src="http://docs.databricks.com/_static/images/clusters/cluster-starting.png"/></a> から <img src="http://docs.databricks.com/_static/images/clusters/cluster-running.png"/></a> へ変更となったら  **<img src="http://docs.databricks.com/_static/images/notebooks/run-all.png"/></a> Run All** をクリック。<br>
-- MAGIC </div>
-- MAGIC 
-- MAGIC 本デモでは操作の流れが分かるようにステップ毎に実行していきます。

-- COMMAND ----------

-- MAGIC %md ### IDEとの連携
-- MAGIC 
-- MAGIC Databricks Connectorというコネクタを通じて、JupyterやRStudioなどのIDEとDatabrikcsのクラスターを接続することが可能です。
-- MAGIC 
-- MAGIC [Databricks Connect — Databricks Documentation](https://docs.databricks.com/dev-tools/databricks-connect.html)
-- MAGIC 
-- MAGIC また、Databrikcs上にRStudio Serverを立ち上げることも可能です。
-- MAGIC 
-- MAGIC [RStudio on Databricks — Databricks Documentation](https://docs.databricks.com/spark/latest/sparkr/rstudio.html)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Databricksデータセットからテーブルを作成する： CSVファイル (by SQL)
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     * {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>

-- COMMAND ----------

-- DBTITLE 1,ストレージ(S3)上にあるCSVファイルの中身を確認
-- MAGIC %fs head /databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv

-- COMMAND ----------

DROP TABLE IF EXISTS diamonds;

CREATE TABLE diamonds
USING csv
OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true", inferSchema "true")

-- COMMAND ----------

-- テーブルを確認
SELECT * from diamonds;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Blobからデータを取得する: Parquetファイル (by Python - pyspark)
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     * {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # 公開Azure Storage Blobから学習データを取得します (WASBプロトコル)
-- MAGIC sourcePath = 'wasbs://mokmok@jixjiadatabricks.blob.core.windows.net/property_data_kanto/'
-- MAGIC 
-- MAGIC df = spark.read\
-- MAGIC           .format('parquet')\
-- MAGIC           .load(sourcePath)\
-- MAGIC           .cache()   #イテレーティブに使うのでメモリーにキャッシュする
-- MAGIC 
-- MAGIC # テーブルとして登録 (Hiveマネージドテーブル)
-- MAGIC df.write.mode('overwrite').saveAsTable('default.property_data_kanto')
-- MAGIC 
-- MAGIC # DataFrameを確認
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC #レコード数を確認
-- MAGIC df.count()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Delta形式にする：　高い信頼性と高性能なデータレイクへ
-- MAGIC 
-- MAGIC ![](/files/shared_uploads/takaaki.yayoi@databricks.com/delta/delta_4.png)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC ##デルタレイクにおけるフォルダを指定
-- MAGIC DELTALAKE_BRONZE_PATH = "dbfs:/home/databricks_demo/delta/property_kanto"
-- MAGIC 
-- MAGIC ##既にフォルダがある場合は削除
-- MAGIC dbutils.fs.rm(DELTALAKE_BRONZE_PATH, recurse=True)
-- MAGIC 
-- MAGIC ##Delta形式でストレージに書き込み
-- MAGIC df.write.format("delta").save(DELTALAKE_BRONZE_PATH)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### パスを指定して参照もできるが・・・

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(
-- MAGIC   spark.sql("select * from delta.`{}` order by 2 desc limit 5".format(DELTALAKE_BRONZE_PATH))
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### パスを指定して参照もできるが、Deltaテーブル化した方がアクセスは簡単

-- COMMAND ----------

DROP TABLE IF EXISTS property_kanto_delta;

CREATE TABLE property_kanto_delta USING DELTA LOCATION 'dbfs:/home/databricks_demo/delta/property_kanto'

-- COMMAND ----------

-- MAGIC %md-sandbox  <span style="color:green"><strong>テーブルを作成しますと、右の [Dataタブ] にも表示されます。</strong></span>

-- COMMAND ----------

SELECT * from default.property_kanto_delta

-- COMMAND ----------

select
  prefecture,
  agebucket,
  count(*)
from
  property_kanto_delta
group by
  1,
  2
order by
  3 desc

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ### データベース一覧

-- COMMAND ----------

-- DBTITLE 0,データベース 一覧
show databases;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### テーブル一覧

-- COMMAND ----------

-- DBTITLE 0,テーブル 一覧
use default;
show tables;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #現在のデータベース名を確認
-- MAGIC spark.catalog.currentDatabase()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 関東物件販売データ
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     * {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <p>今回の学習データはすでにクレンジングと加工済みですので、そのまま使ってモデルを作成することも可能です（後述）。<br>
-- MAGIC ETLやデータの加工整形に関してDatabricks実践シリーズ<a href='https://jixjia.com/ja/handson-learning-databricks/#session1'>第一回</a>と<a href='https://jixjia.com/ja/handson-learning-databricks/#session2'>第二回</a>の内容をご参考下さい。</p>
-- MAGIC <br>
-- MAGIC <h2>特徴量説明</h2>
-- MAGIC <ul>
-- MAGIC   <li>PropertyName: 物件名 or 建物名</li>
-- MAGIC   <li>Price: 売却価格 (単位：万円)</li>
-- MAGIC   <li>Prefecture: 物件所在都道府県</li>
-- MAGIC   <li>Municipality: 物件市区町村</li>
-- MAGIC   <li>Address: 物件住所 (プライバシー保護の為番地以降は省略)</li>
-- MAGIC   <li>FloorPlan: 間取り (例：1LDK, 3LDK+S)</li>
-- MAGIC   <li>PublicTransportName: 最寄り駅路線名 (例：JR中央線、都営三田線)</li>
-- MAGIC   <li>NearestStation: 最寄り駅名</li>
-- MAGIC   <li>DistanceMethod: 最寄り駅へのアクセス方法 (例：徒歩、バス、車)</li>
-- MAGIC   <li>Distance_min: 最寄り駅へのアクセス時間 (単位：分)</li>
-- MAGIC   <li>FloorSpace_m2: 専有面積 (単位：平米)</li>
-- MAGIC   <li>OtherSpace_m2: バルコニー面積 (単位：平米)</li>
-- MAGIC   <li>BuiltYear: 築年 (例：1998、2018)</li>
-- MAGIC </ul>
-- MAGIC <span style='color:green;'>
-- MAGIC 人工的に作られた特徴量：
-- MAGIC <ul>
-- MAGIC   <li>Age: 築年数 (BuiltYearから算出)</li>
-- MAGIC   <li>FloorSpaceBucket: 専有面積をカテゴリ化 (10段階)</li>
-- MAGIC   <li>AgeBucket: 築年数をカテゴリ化 (8段階)</li>
-- MAGIC   <li>DistanceBucket: 最寄り駅アクセス分数をカテゴリ化 (21段階)</li>
-- MAGIC </ul>
-- MAGIC </span>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 探索的データ解析 (EDA)
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     * {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <p>まず異なるアングルから特徴量と目標変数(Price)間の相関性、分布性質、線形関係など探索し、理解を深めていきます。<br>
-- MAGIC ここでは<span style="color:#38a">SQL</span>と<span style="color:#38a">ドラッグ&amp;ドロップ</span>のみでEDAを行っています:</p>
-- MAGIC 
-- MAGIC <h3>一部解析の例</h3>
-- MAGIC <div style="padding:10px 0;">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-age-bucket.gif" width="700px">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-age-count.gif" width="700px">
-- MAGIC </div>
-- MAGIC <div style="padding:10px 0;">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-distance.gif" width="700px">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-floor-space.gif" width="700px">
-- MAGIC </div>
-- MAGIC <div style="padding:10px 0;">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-municipality.gif" width="700px">
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-distance-method.gif" width="700px">
-- MAGIC </div>
-- MAGIC 
-- MAGIC <div style="padding:10px 0;">
-- MAGIC   <h2>目標変数の密度分布 (Bin=50)</h2>
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/dbfs-mnt/projects/suumo-property-price-assessment/price-distribution.gif" width="1200px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### SQLを使ったデータ可視化

-- COMMAND ----------

-- DBTITLE 0,SQLを使ったデータ可視化
-- MAGIC %sql
-- MAGIC SELECT * FROM property_kanto_delta

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 数値型特徴量の統計情報

-- COMMAND ----------

-- DBTITLE 0,数値型特徴量の統計情報
-- MAGIC %python
-- MAGIC display(
-- MAGIC   sql('Select Price, Distance_min, FloorSpace_m2, OtherSpace_m2, Age From property_data_kanto')
-- MAGIC   .summary()
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Deltaファイルの実体は？

-- COMMAND ----------

-- MAGIC %fs ls dbfs:/home/databricks_demo/delta/property_kanto

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### レコードをDelete
-- MAGIC (通常のデータレイクでは非対応な処理)

-- COMMAND ----------

-- 千葉県を削除
DELETE FROM property_kanto_delta
WHERE Prefecture = '千葉県';


-- データを確認
SELECT * FROM property_kanto_delta;


-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Deltaのメタデータ

-- COMMAND ----------

DESCRIBE detail property_kanto_delta

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Deltaの履歴

-- COMMAND ----------

DESCRIBE history property_kanto_delta

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### タイムトラベルを実行

-- COMMAND ----------

SELECT * FROM property_kanto_delta Version AS of 1

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <h1>次のコマンドでは、データ操作を行い結果をグラフで表示します。</h1>  
-- MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
-- MAGIC (1) "都道府県名：Prefecture"、"市区町村：Municipality"、"価格：price" というカラムを選択して、都道府県/市区町村別の平均価格で算出して、価格にてソート。<br>
-- MAGIC (2) テーブル結果を参照。<br>
-- MAGIC (3) テーブルの下部のバーチャートをクリック <img src="http://docs.databricks.com/_static/images/notebooks/chart-button.png"/></a> アイコン<br>
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 全件を参照

-- COMMAND ----------

-- DBTITLE 0,全件を参照
SELECT
  Prefecture,
  Municipality,
  avg(Price) AS proce
FROM
  property_kanto_delta
GROUP BY
  1,
  2
ORDER BY
  3 desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### TOP20を参照

-- COMMAND ----------

-- DBTITLE 0,TOP20を参照
SELECT
  Prefecture,
  Municipality,
  avg(Price) AS proce
FROM
  property_kanto_delta
GROUP BY
  1,
  2
ORDER BY
  3 desc
limit
  20

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 同じオペレーションを Python DataFrame API にて実施してみる！  
-- MAGIC 
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
-- MAGIC (1) こちらはSQLノートブックです。デフォルト言語はSQLとなっています。<br>
-- MAGIC (2) pythonを利用するには、 `%python` というマジックコマンドを利用します。<br>
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import avg,desc
-- MAGIC 
-- MAGIC display(
-- MAGIC   df.select("Prefecture","Municipality","Price")
-- MAGIC   .groupBy("Prefecture","Municipality")
-- MAGIC   .agg(avg("Price").alias("Price"))
-- MAGIC   .sort(desc("Price"))
-- MAGIC )

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## その他
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
-- MAGIC (1) コラボレーション機能：　他の人のNotobookを参照してみる。<br>
-- MAGIC (2) コメントを記述してみる。<br>
-- MAGIC (3) 改訂履歴を見てみる。<br>
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Notebookでのコラボレーション
-- MAGIC <p  style="position:relative;width: 1000px;height: 450px;overflow: hidden;">
-- MAGIC <img style="margin-top:25px;" src="https://psajpstorage.blob.core.windows.net/commonfiles/Collaboration001.gif" width="1000">
-- MAGIC </p>
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 最後に、ストリーミングを試してみる

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC <h2> ![](https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-tiny-logo.png) バッチ処理とストリーミング処理の統合</h2>
-- MAGIC ストリーミング処理とバッチ処理が多重で動いています。 (inserts and reads)
-- MAGIC * `property` テーブルにストリーミング処理を実行
-- MAGIC * 同じく`property` テーブルに対して `INSERT` 処理を10秒おきに実行
-- MAGIC * 補足： `writeStream` を使用も可能であるが、コミュニティエディションでの実行を想定して利用していません
-- MAGIC 
-- MAGIC ### Databricks デルタアーキテクチャ
-- MAGIC Databricks Deltaを使ってストリーム処理とバッチを融合させた新しいELTアーキテクチャの実装を行います。<br>また、ストリーム処理にSpark Structure Streamingを利用し、バッチと同じ手法でデータを処理したり、クエリをかけたりします。終いには外部のBIツール(Power BIを予定)と連携して、リアルタイムのデータ分析を行って見たいと思います。
-- MAGIC 
-- MAGIC <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta_architecture_demo.gif" width="85%">

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 簡易化のために都道府県別のサマリーテーブルを作成

-- COMMAND ----------

-- MAGIC %python
-- MAGIC from pyspark.sql.functions import avg,desc
-- MAGIC 
-- MAGIC # 公開Azure Storage Blobから学習データを取得します (WASBプロトコル)
-- MAGIC sourcePath = 'wasbs://mokmok@jixjiadatabricks.blob.core.windows.net/property_data_kanto/'
-- MAGIC 
-- MAGIC tempDF = spark.read\
-- MAGIC           .format('parquet')\
-- MAGIC           .load(sourcePath)\
-- MAGIC           .cache()   #イテレーティブに使うのでメモリーにキャッシュする
-- MAGIC 
-- MAGIC # テーブルとして登録 (Hiveマネージドテーブル)
-- MAGIC summaryDF = tempDF.select("Prefecture").groupBy("Prefecture").count()
-- MAGIC summaryDF.write.mode('overwrite').saveAsTable('default.property_summary')
-- MAGIC 
-- MAGIC display(summaryDF.sort(desc("count")))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### 都道府県別のデルタテーブルを作成 

-- COMMAND ----------

-- MAGIC %python
-- MAGIC ##デルタレイクにおけるのフォルダを指定
-- MAGIC DELTALAKE_SILVER_PATH = "dbfs:/home/databricks_demo/delta/property_summary"
-- MAGIC 
-- MAGIC ##既にフォルダがある場合は削除
-- MAGIC dbutils.fs.rm(DELTALAKE_SILVER_PATH, recurse=True)
-- MAGIC 
-- MAGIC ##Delta形式で書き込み
-- MAGIC summaryDF.write.format("delta").save(DELTALAKE_SILVER_PATH)

-- COMMAND ----------

DROP TABLE IF EXISTS property_summary_delta;

CREATE TABLE property_summary_delta USING DELTA LOCATION '/home/databricks_demo/delta/property_summary'

-- COMMAND ----------

select
  Prefecture,
  count
from
  property_summary_delta
order by
  2 desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### テストのために`神奈川県`のデータを削除します

-- COMMAND ----------

delete from
  property_summary_delta
where
  Prefecture = '神奈川県';

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### `readStream`でストリーミング処理を実施

-- COMMAND ----------

-- MAGIC %python
-- MAGIC property_readStream = spark.readStream.format("delta").load(DELTALAKE_SILVER_PATH)
-- MAGIC property_readStream.createOrReplaceTempView("property_readStream")

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### ストリーミング状況を検索します。
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC (1)　初期状態では`神奈川県`がありません。<br/>
-- MAGIC (2)　１つ下の`Insert`処理を実施すると`神奈川県`がグラフに登場します。<br/>

-- COMMAND ----------

select
  Prefecture,
  count
from
  property_readStream

-- COMMAND ----------

-- MAGIC %python
-- MAGIC import time
-- MAGIC i = 1
-- MAGIC while i <= 10:
-- MAGIC   # Insert処理の実行(神奈川県のデータを追加)
-- MAGIC   insert_sql = "INSERT INTO property_summary_delta VALUES ('神奈川県', 1500)"
-- MAGIC   spark.sql(insert_sql)
-- MAGIC   print('property_summary_delta: inserted new row of data, loop: [%s]' % i)
-- MAGIC     
-- MAGIC   # ループ
-- MAGIC   i = i + 1
-- MAGIC   time.sleep(3)

-- COMMAND ----------

-- MAGIC %md-sandbox  <span style="color:green"><strong>最後にストリーミング処理をキャンセルします。</strong></span>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## タイムトラベルを実施
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
-- MAGIC 
-- MAGIC (1)　WRITEが行われていることを確認できます。<br/>
-- MAGIC (2)　WRITE以前のテーブル状態を参照してみましょう。<br/>

-- COMMAND ----------

DESCRIBE HISTORY property_summary_delta

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC WRITE以前のデータを参照すると、神奈川県データが存在しないことがわかります。

-- COMMAND ----------

-- DBTITLE 0,神奈川県データがありません
select
  Prefecture,
  count
from
  property_summary_delta VERSION AS OF 1
order by
  2 desc

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # END
-- MAGIC <head>
-- MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
-- MAGIC   <style>
-- MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
-- MAGIC   </style>
-- MAGIC </head>
