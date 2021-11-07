# Databricks notebook source
# MAGIC %md # Databricksの基礎
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC 
# MAGIC 1. マジックコマンド
# MAGIC 1. Sparkの基礎
# MAGIC   1. Python
# MAGIC   1. SQL
# MAGIC   1. R
# MAGIC 1. Delta Lake
# MAGIC 1. Streaming
# MAGIC 1. 機械学習

# COMMAND ----------

# DBTITLE 1,(デモ用準備) - ユニークなパス名を設定(ユーザー間の衝突回避)
import re

# Username を取得。
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Usernameをファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '_', username_raw).lower()
dbname = f'tutorial_db_{username}'

print(f'>>> username => {username}')
print(f'>>> dbname => {dbname}')

# 使用するDatabaseを固定する
sql(f'CREATE DATABASE IF NOT EXISTS {dbname}')
sql(f'use {dbname};')

# COMMAND ----------

# MAGIC %md ## マジックコマンド
# MAGIC ----
# MAGIC 
# MAGIC * [%fs マジック コマンドを使用する](https://docs.microsoft.com/ja-jp/azure/databricks/data/databricks-file-system#dbfs-dbutils-fs-magic-commands)

# COMMAND ----------

# DBTITLE 1,Object Storage(S3, blob, GCS)のリスト
# MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/csv/ggplot2/

# COMMAND ----------

# DBTITLE 1,ファイルのhead
# MAGIC %fs head /databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv

# COMMAND ----------

# DBTITLE 1,シェルコマンドの実行(Driver node上のbashで実行される)
# MAGIC %sh 
# MAGIC ls -l

# COMMAND ----------

# MAGIC %md ## マジックコマンド(コード上)
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC マジックコマンドをコード上(Python/Scala/R)から呼ぶ場合は`dbutils`モジュールを使用します。
# MAGIC 
# MAGIC * [Databricks ユーティリティ](https://docs.microsoft.com/ja-jp/azure/databricks/dev-tools/databricks-utils)

# COMMAND ----------

# MAGIC %python
# MAGIC display(
# MAGIC   dbutils.fs.ls('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/')
# MAGIC )

# COMMAND ----------

# MAGIC %scala
# MAGIC display(
# MAGIC   dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/")
# MAGIC )

# COMMAND ----------

# MAGIC %r
# MAGIC dbutils.fs.ls("/databricks-datasets/Rdatasets/data-001/csv/ggplot2/")

# COMMAND ----------

# MAGIC %python
# MAGIC dbutils.fs.help()

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC class(iris)
# MAGIC class(distributed_iris)

# COMMAND ----------

# MAGIC %md ## Sparkの基礎(データレイクとして使い方)
# MAGIC 
# MAGIC -----
# MAGIC ファイルフォーマットに加えて、RDBMSやNoSQLなどからデータを入出力できます。
# MAGIC 対応フォーマット・データソースは以下のドキュメントにまとまっています。
# MAGIC (サンプルコード・notebookも提供されています。)
# MAGIC 
# MAGIC * [Data Soruce](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/)

# COMMAND ----------

# DBTITLE 1,サンプルCSVデータ
# MAGIC %fs head /databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv

# COMMAND ----------

# MAGIC %md ### Python(PySpark)
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * [Python 開発者向けのDatabricks](https://docs.microsoft.com/ja-jp/azure/databricks/languages/python)

# COMMAND ----------

# DBTITLE 1,CSVファイルをSpark DataFrameとして読み込む
df = (
  spark.read.format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .load('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv') 
)

display(df)

# COMMAND ----------

# DBTITLE 1,簡単なETL
df_cleaned = (
  df
  .where('price between 2000 and 2500') # カラムの条件
  .withColumn('magic_number', df.x * df.y + df.z ) # カラムの追加
  .select('carat', 'cut', 'color', 'clarity', 'price', 'magic_number') # カラムの抽出
)

display( df_cleaned )

# COMMAND ----------

# DBTITLE 1,JSONファイルとして書き出す
( 
  df_cleaned.write
  .format('json')
  .mode('overwrite')
  .save(f'/tmp/{username}/diamonds_json')
)

# COMMAND ----------

# DBTITLE 1,書き出されたJSONファイルの確認1
display( dbutils.fs.ls(f'/tmp/{username}/diamonds_json') )

# COMMAND ----------

# DBTITLE 1,書き出されたJSONファイルの確認2
# MAGIC %fs head dbfs:<ここを書き換えてください>

# COMMAND ----------

# MAGIC %md ### (補足)従来のデータレイクの制限と限界
# MAGIC 
# MAGIC -----
# MAGIC * ファイルのパーティションをユーザーが管理しないといけない
# MAGIC * 細かいファイルがどんどん増えていく
# MAGIC * ファイル数が増えるにつれて読み込みに時間がかかる
# MAGIC * レコードは追記のみ(UPDATE, DELETE, MERGEができない)
# MAGIC * スキーマの整合性はユーザー側でチェックしないといけない
# MAGIC * 検索条件がパーティションキーでない場合、全てのファイルを開く必要がある
# MAGIC * Indexingなどの最適化機能がない
# MAGIC 
# MAGIC 
# MAGIC など。
# MAGIC 
# MAGIC 上記の課題は**Delta Lake**によって解消します。

# COMMAND ----------

# MAGIC %md ### SQL
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC SQLではDataへのアクセスのため、"テーブル"名、もしくは、"View"名でアクセスします。
# MAGIC Databricks(Spark)でSQLを使用するには以下の2つのアプローチがあります。
# MAGIC 
# MAGIC 1. Spark Daraframeから`createOrReplaceTempView()`でTempViewを定義する
# MAGIC   - 一時的、同セッション内で有効
# MAGIC 1. `CREATE TABLE`クエリなどを使用してHiveメタストアにテーブル・Viewを登録する
# MAGIC   - 永続的
# MAGIC   - 同workspace内から参照可能
# MAGIC   - アクセス制限可能
# MAGIC   - メニュー上にあるデータカタログ("DATA")から参照可能)
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC また、SQLの実行は以下の2つの方法があります。
# MAGIC 
# MAGIC 1. SQLセル内でSQLクエリを書いて実行(クエリの結果が直接出力される)
# MAGIC 2. コード内で`sql()`にSQLクエリを引数で渡してを実行する(結果はSpark Dataframeで返される)
# MAGIC 
# MAGIC ------
# MAGIC 
# MAGIC * [SQL 開発者向けのDatabricks](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/spark-sql/)
# MAGIC * [SQL Reference](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/spark-sql/)

# COMMAND ----------

# DBTITLE 1,Spark DataFrameからTempviewを作成する
df = (
  spark.read.format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .load('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv') 
)
df.createOrReplaceTempView('tempview_diamonds')

# COMMAND ----------

# DBTITLE 1,SQLセルで実行
# MAGIC %sql
# MAGIC SELECT 
# MAGIC   carat, cut, color, clarity, price, x * y + z as magic_number
# MAGIC FROM tempview_diamonds

# COMMAND ----------

# DBTITLE 1,コード内のsql()で実行
df_sql = sql('''
  SELECT 
    carat, cut, color, clarity, price, x * y + z as magic_number
  FROM tempview_diamonds
''')

display( df_sql )

# COMMAND ----------

# DBTITLE 1,Hiveメタストアにテーブルを登録する
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE table_diamonds
# MAGIC USING csv
# MAGIC OPTIONS (path "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv", header "true")

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM table_diamonds;

# COMMAND ----------

# MAGIC %md ### SparkR
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC Rによるデータの読み込み、Spark DataFrameの扱い方。
# MAGIC 
# MAGIC * [R 開発者向けのDatabricks](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/sparkr/)
# MAGIC   - [SparkR の概要](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/sparkr/overview)
# MAGIC   - [SparkR ML のチュートリアル](https://docs.microsoft.com/ja-jp/azure/databricks/spark/latest/sparkr/tutorials/)
# MAGIC   
# MAGIC * [Databricks Academy](https://academy.databricks.com/) - Free Customer Learning
# MAGIC 
# MAGIC   - Databricks with R

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC df <- SparkR::read.df(
# MAGIC   "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv",
# MAGIC   source = "csv",
# MAGIC   header = TRUE,
# MAGIC   inferSchema = TRUE
# MAGIC )
# MAGIC 
# MAGIC display(df)

# COMMAND ----------

# MAGIC %r
# MAGIC display(iris)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # R dataframeからSpark Dataframeに変換
# MAGIC distributed_iris <- SparkR::createDataFrame(iris)
# MAGIC display(distributed_iris)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC # tempview_diamondsはpythonコード上で定義したTempView
# MAGIC df_ret = SparkR::sql("SELECT * FROM tempview_diamonds")
# MAGIC display(df_ret)

# COMMAND ----------

# MAGIC %r
# MAGIC 
# MAGIC print( class(iris) ) # "data.frame" <= Rのdataframe
# MAGIC print( class(distributed_iris) ) # "SparkDataFrame" <= Sparkのdataframe

# COMMAND ----------

# MAGIC %md ## Delta Lake
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">
# MAGIC <hr>
# MAGIC <h3>データレイクに<span style="color='#38a'">信頼性</span>と<span style="color='#38a'">パフォーマンス</span>をもたらす</h3>
# MAGIC <p>本編はローン審査データを使用してDelta LakeでETLを行いながら、その主要機能に関して説明していきます。</p>
# MAGIC <div style="float:left; padding-right:60px; margin-top:20px; margin-bottom:200px;">
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta-lake-square-black.jpg" width="220">
# MAGIC </div>
# MAGIC 
# MAGIC <div style="float:left; margin-top:0px; padding:0;">
# MAGIC   <h3>信頼性</h3>
# MAGIC   <br>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC     <li>次世代データフォーマット技術</li>
# MAGIC     <li>トランザクションログによるACIDコンプライアンス</li>
# MAGIC     <li>DMLサポート（更新、削除、マージ）</li>
# MAGIC     <li>データ品質管理　(スキーマージ・エンフォース)</li>
# MAGIC     <li>バッチ処理とストリーム処理の統合</li>
# MAGIC     <li>タイムトラベル (データのバージョン管理)</li>
# MAGIC    </ul>
# MAGIC 
# MAGIC   <h3>パフォーマンス</h3>
# MAGIC   <br>
# MAGIC   <ul style="padding-left: 30px;">
# MAGIC      <li>スケーラブルなメタデータ</li>
# MAGIC     <li>コンパクション (Bin-Packing)</li>
# MAGIC     <li>データ・インデックシング</li>
# MAGIC     <li>データ・スキッピング</li>
# MAGIC     <li>ZOrderクラスタリング</li>
# MAGIC     <li>ストリーム処理による低いレイテンシー</li>
# MAGIC   </ul>
# MAGIC </div>
# MAGIC 
# MAGIC <div style="display:block; clear:both; padding-top:20px;">
# MAGIC   <div style="background: #ff9; margin-top:10px;">
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/exclamation-yellow.png" width="25px"><span style="padding-left:10px; padding-right:15px;">注) セル12、14と42は意図的にエラーを起こすよう作成しています</span>
# MAGIC </div>
# MAGIC </div>
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC * [Delta Engine および Delta Lake ガイド](https://docs.microsoft.com/ja-jp/azure/databricks/delta/)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### サンプルデータ
# MAGIC 
# MAGIC 今回使用するデータはLending Clubが公開している2012年から2017年までの融資審査データです。
# MAGIC ローン申請者情報(匿名)、現在のローンステータス(延滞、全額支払い済みなど)などが含まれます。
# MAGIC 
# MAGIC **データ辞書** https://www.kaggle.com/wendykan/lending-club-loan-data <br>
# MAGIC 
# MAGIC <!--<img src="https://jixjiastorage.blob.core.windows.net/public/datasets/lending-club-loan/data_sample.png" width="95%">-->

# COMMAND ----------

# DBTITLE 1,データの読み込み
# パス指定
source_path = 'dbfs:/databricks-datasets/samples/lending_club/parquet/'
delta_path = f'dbfs:/home/{username}/delta/lending-club-loan/'

# 既存のデータを削除
dbutils.fs.rm(delta_path, recurse=True)

# データを読み込む
df = spark.read.parquet(source_path)

# レコード数
print(df.count())

# randomSplit()を使って、5%のサンプルを読み取る
(data, data_rest) = df.randomSplit([0.05, 0.95], seed=123)

# 読み込まれたデータを参照
display( data )

# COMMAND ----------

# DBTITLE 1,(参考: DS向け) データのサマリを見る1 - dbutils.data.summarize 
# データソースはparquetファイル(slow)
dbutils.data.summarize(df)

# COMMAND ----------

# DBTITLE 1,(参考: DS向け) データのサマリを見る2 - pandas_profiling
from pandas_profiling import ProfileReport
df_profile = ProfileReport(df.toPandas(), minimal=True, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------

# DBTITLE 1,データの準備/Deltaで保存
# 簡単な処理を行い、結果をParquetとして保存
from pyspark.sql.functions import col, expr

data.select('loan_amnt', 
            'term',
            'int_rate',
            'grade',
            'addr_state',
            'emp_title',
            'home_ownership',
            'annual_inc',
            'loan_status')\
  .withColumn('int_rate', expr('cast(replace(int_rate,"%","") as float)'))\
  .withColumnRenamed('addr_state', 'state')\
  .write\
  .format('delta')\
  .mode('overwrite')\
  .save(delta_path)

# COMMAND ----------

# DBTITLE 1,テーブル化(Deltaテーブル)
# Databaseを作成
sql(f'CREATE DATABASE IF NOT EXISTS {dbname}')
sql(f'USE {dbname}')

# テーブルとして登録
sql(f'DROP TABLE IF EXISTS LBS')
sql(f'CREATE TABLE LBS USING delta LOCATION "{delta_path}"')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:10px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-traditional.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>データの読み込み(クエリ)</h3>
# MAGIC   Parquetによる分散ファイルは<span style="color:green"><strong>効率よく読む</strong></span>事ができます</span>.
# MAGIC </div>

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC -- データをSELECT文でクエリしてみよう
# MAGIC Select state, loan_status, count(*) as counts 
# MAGIC From LBS  
# MAGIC Group by state, loan_status
# MAGIC Order by counts desc

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:10px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>Deltaの物理的な構造</h3>
# MAGIC   Delta LakeはParquet技術の上に成り立っています
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Deltaの中身を見てみよう
# %fs ls /home/{username}/delta/lending-club-loan/

display( 
  dbutils.fs.ls(f'{delta_path}') # <= Deltaを書き込んだディレクトリをリストしてみる
)

# COMMAND ----------

# DBTITLE 1,トランザクションログの中身を見てみよう
# %fs head /home/parquet/lending-club-loan/_delta_log/00000000000000000000.json

dbutils.fs.head(f'{delta_path}/_delta_log/00000000000000000000.json')

# COMMAND ----------

# DBTITLE 1,データの変更管理
# MAGIC %sql
# MAGIC -- Describe History機能でデータの変更履歴を監査
# MAGIC Describe History LBS

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:20px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>DMLサポート</h3>
# MAGIC   ここで従来のデータレイクでは出来なかった<br>データの<span style="color:green"><strong>更新</strong></span>、<span style="color:green"><strong>マージ</strong></span>や<span style="color:green"><strong>削除</strong></span>などのETL操作を行ってみたいと思います
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,現在のデータを把握
# MAGIC %sql
# MAGIC -- マップで州ごとの融資総額を見てみよう
# MAGIC Select *
# MAGIC From LBS

# COMMAND ----------

# DBTITLE 1,削除を行う (従来のデータレイクでは不可)
# MAGIC %sql
# MAGIC -- 'WA'州を削除する
# MAGIC Delete From LBS Where State = 'WA';
# MAGIC 
# MAGIC -- データを見る
# MAGIC Select * 
# MAGIC From LBS

# COMMAND ----------

# DBTITLE 1,更新をする (従来のデータレイクでは不可)
# MAGIC %sql
# MAGIC -- 'NY'州のローン申込額を20倍にする!
# MAGIC Update LBS Set loan_amnt = loan_amnt * 20 Where State = 'NY';
# MAGIC 
# MAGIC -- データを見る
# MAGIC Select *
# MAGIC From LBS 

# COMMAND ----------

# DBTITLE 1,データのマージ
# MAGIC %md-sandbox 
# MAGIC <div style="float:left; padding-right:40px;">
# MAGIC   <h4>従来のデータレイク (7 ステップ)</h4>
# MAGIC   <ol style="padding-left: 20px;">
# MAGIC     <li>更新するデータを取得 (新規データ)</li>
# MAGIC     <li>更新されるデータを取得 (既存データ)</li>
# MAGIC     <li>更新されないデータを取得 (既存データ)</li>
# MAGIC     <li>それぞれのTempテーブルを用意</li>
# MAGIC     <li>既存データを全て削除(分散物理ファイルも含めて)</li>
# MAGIC     <li>一つのテーブルへ合併し、元のテーブル名へ再命名</li>
# MAGIC     <li>Tempテーブルをドロップ</li>
# MAGIC    </ol>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <img src="https://pages.databricks.com/rs/094-YMS-629/images/merge-into-legacy.gif" width="800px">
# MAGIC </div>
# MAGIC <div style="clear:both; display:block;">
# MAGIC   <h4>Delta Lake (2 ステップ)</h4>
# MAGIC   <ol style="padding-left: 20px;">
# MAGIC     <li>更新するデータを取得 (新規データ)</li>
# MAGIC     <li>`MERGE INTO`実行</li>
# MAGIC    </ol>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,マージデータを用意
# Demo用のマージするデータを別途作成 (本番環境では新規データがこれに値する)
columns = sql('Describe Table LBS').filter('data_type != ""').select('col_name').rdd.flatMap(lambda x: x).collect()

merge_df = sc.parallelize([
   [999999, '36 months', 5.00, 'A', 'IA', 'Demo Data', 'RENT', 1000000, 'Current']
]).toDF(columns)

merge_df.createOrReplaceTempView("merge_table")
display(merge_df)

# COMMAND ----------

# DBTITLE 1,マージする (従来のデータレイクではMergeクエリは不可)
# MAGIC %sql
# MAGIC -- マージオペレーションを行う
# MAGIC Merge Into LBS as target
# MAGIC Using merge_table as source
# MAGIC on target.State = source.State
# MAGIC when MATCHED Then Update Set *
# MAGIC When Not MATCHED Then Insert *
# MAGIC ;
# MAGIC 
# MAGIC -- データを見る
# MAGIC Select *
# MAGIC From LBS 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:20px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>データ品質管理がらくらく</h3>
# MAGIC   スキーマエンフォースで既存のスキーマと一致しないDML操作を<span style="color:green"><strong>拒否</strong></span><br>もしくはスキーマエボリューションでスキーマの差分を<span style="color:green"><strong>完璧にマージ</strong></span>する事ができる
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,スキーマが変わったデータを用意
# 既存のデータの10行を読み込んで、新たなカラムを追加
new_df = spark.read\
              .format('delta')\
              .load(delta_path)\
              .limit(10)\
              .withColumn('interest_flag', expr('case when int_rate >=6 then "high" else "low" end'))

display(new_df)

# COMMAND ----------

# DBTITLE 1,既存データへ追加 (スキーマが合わない)
# スキーマが違うデータセットへ書き込む (Append)
new_df.write.format('delta').mode('append').save(delta_path)

# COMMAND ----------

# MAGIC %md  今度は`mergeSchema`オプションを使って異なるスキーマ同士のデータを融合させましょう（スキーマエボリューション)

# COMMAND ----------

# DBTITLE 1,既存データへ追加 (スキーマが融合される)   :(従来のデータレイクでは不可)
# スキーマが違うデータセットへ書き込む (with スキーマエボリューション)
new_df.write.format('delta').option('mergeSchema','true').mode('append').save(delta_path)

# COMMAND ----------

# MAGIC %sql
# MAGIC describe LBS

# COMMAND ----------

# MAGIC %sql
# MAGIC -- データを見る
# MAGIC Select *
# MAGIC From LBS 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:20px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>タイムトラベルで過去へ戻ろうく</h3>
# MAGIC   Delta Lakeのログを利用して<span style="color:green"><strong>以前のデータスナップショット</strong></span>を取得したり、<br>データ<span style="color:green"><strong>変更履歴の監査</strong></span>をしたり、<span style="color:green"><strong>ロールバック</strong></span>をすることが容易に出来ます
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,もう一度データ変更管理を見てみよう (従来のデータレイク・DWHでは不可)
# MAGIC %sql
# MAGIC -- Describe History機能でデータの変更履歴を監査
# MAGIC Describe History LBS

# COMMAND ----------

# DBTITLE 1,過去へ戻ろう (その一)  : (従来のデータレイク・DWHでは不可)
# MAGIC %sql
# MAGIC -- バージョンを指定してスナップショットを取得
# MAGIC Select * 
# MAGIC From LBS Version AS OF 2

# COMMAND ----------

# DBTITLE 1,過去へ戻ろう (その二)  : (従来のデータレイク・DWHでは不可)
# 時間をしてしてスナップショットを取得
desiredTimestamp = spark.sql("Select timestamp From (Describe History LBS) Order By timestamp Desc").take(10)[-1].timestamp

print(f'desiredTimestamp => {desiredTimestamp}')

display(
  sql(f"Select * From LBS TIMESTAMP AS OF '{desiredTimestamp}'")
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div style="float:left; padding-right:10px; margin-top:20px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>OPTIMIZEで性能を向上させるく</h3>
# MAGIC   小さなファイルを合併してくれる<span style="color:green"><strong>Optimize</strong></span>コマンドをはじめ<br>、<span style="color:green"><strong>Data Skpping</strong></span>,<span style="color:green"><strong>ZOrder Clustering</strong></span>などの機能をを利用して、<br>クエリ検索性能の向上を容易に実現します。
# MAGIC </div>

# COMMAND ----------

# MAGIC %sql
# MAGIC -- OPTIMIZEでデータをコンパクト化、インデックス作成。更にZORDERで良く使われるカラムのデータを物理的に一カ所へまとめる
# MAGIC OPTIMIZE LBS ZORDER By (state)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- セル10と全く同じクエリを実行
# MAGIC Select state, loan_status, count(*) as counts 
# MAGIC From LBS
# MAGIC Group by state, loan_status
# MAGIC Order by counts desc

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC 
# MAGIC ## Structured Streaming 
# MAGIC 
# MAGIC <div style="float:left; padding-right:10px; margin-top:20px;">
# MAGIC   <img src='https://jixjiadatabricks.blob.core.windows.net/images/data-lake-no-label.png' width='70px'>
# MAGIC </div>
# MAGIC <div style="float:left;">
# MAGIC   <h3>BatchとStreamingの統合</h3>
# MAGIC   Delta Lake/SparkはStraeming処理をバッチ処理と同等に扱えます。ここでは、WikipediaのIRCチャットの内容を受信しているKafkaサーバからリアルタイムでデータを読み込むサンプルを見ていきましょう。
# MAGIC   
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta_architecture_demo.gif" width="1200px">
# MAGIC 
# MAGIC    <h4>Kafkaサーバー情報</h4>
# MAGIC   <table>
# MAGIC     <tr><td>オプション名</td><td>kafka.bootstrap.servers</td></tr>
# MAGIC     <tr><td>サーバー名</td><td>server2.databricks.training:9092</td></tr>
# MAGIC     <tr><td>トピック</td><td>en</td></tr>
# MAGIC   </table>
# MAGIC   <br>
# MAGIC   <h4>JSONストリーム</h4>
# MAGIC   <p>DatabricksではWikipediaの公式IRCチャネルから編集データを受信し、JSONに変換して自社Kafkaサーバーへ送信しています。
# MAGIC     <br>今回はこのKafkaサーバーに流れ込んだ英語の記事(トピック)にサブスクライブし、リアルタイムで処理していきます。

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col
import re

# ファイルパスの設定(書き込み先など)
## Username を取得。
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
## Username の英数字以外を除去し、全て小文字化。Usernameをファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '_', username_raw).lower()

homeDir = f'/home/{username}/streaming/wikipedia/'
bronzePath = homeDir + "bronze.delta"
bronzeCkpt = homeDir + "bronze.checkpoint"

# 保存先をリセット
dbutils.fs.rm(homeDir, True)

#　JSONデータのスキーマ定義
schema = StructType([
  StructField("channel", StringType(), True),
  StructField("comment", StringType(), True),
  StructField("delta", IntegerType(), True),
  StructField("flag", StringType(), True),
  StructField("geocoding", StructType([
    StructField("city", StringType(), True),
    StructField("country", StringType(), True),
    StructField("countryCode2", StringType(), True),
    StructField("countryCode3", StringType(), True),
    StructField("stateProvince", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
  ]), True),
  StructField("isAnonymous", BooleanType(), True),
  StructField("isNewPage", BooleanType(), True),
  StructField("isRobot", BooleanType(), True),
  StructField("isUnpatrolled", BooleanType(), True),
  StructField("namespace", StringType(), True),         
  StructField("page", StringType(), True),              
  StructField("pageURL", StringType(), True),           
  StructField("timestamp", StringType(), True),        
  StructField("url", StringType(), True),
  StructField("user", StringType(), True),              
  StructField("userURL", StringType(), True),
  StructField("wikipediaURL", StringType(), True),
  StructField("wikipedia", StringType(), True),
])

#　JSONストリームを解析し、Deltaに保存
input_DF = (
  spark
  .readStream
  .format('kafka')                          # Kafkaをソースと指定
  .option('kafka.bootstrap.servers', 
          'server2.databricks.training:9092')
  .option('subscribe', 'en')
  .load()
)

# ELTをして、Deltaに書き込む
(
  input_DF
  .withColumn('json', from_json(col('value').cast('string'), schema))   # Kafkaのバイナリデータを文字列に変換し、from_json()でJSONをパース
  .select(col("json.*"))                    # JSONの子要素だけを取り出す
  .writeStream                              # writeStream()でストリームを書き出す
  .format('delta')                          # Deltaとして保存
  .option('checkpointLocation', bronzeCkpt) # チェックポイント保存先を指定
  .outputMode('append')                     # マイクロバッチの結果をAppendで追加
  .queryName('Bronze Stream')               # ストリームに名前を付ける（推奨）
  .start(bronzePath)                        # start()でストリーム処理を開始 (アクション)
)

# COMMAND ----------

# MAGIC %md ### 以下の2つのコードの違いはどこにありますか?

# COMMAND ----------

# DBTITLE 1,バッチ型DataFrameとして表示
# データフレームの確認
df = spark.read.format('delta').load(bronzePath)

display( df )

# COMMAND ----------

# DBTITLE 1,Stream型DataFrameとして表示
# データフレームの確認2
df = spark.readStream.format('delta').load(bronzePath)

display( df )

# COMMAND ----------

# MAGIC %md ### StreamingはSQLからも参照可能です

# COMMAND ----------

spark.readStream.format('delta').load(bronzePath).createOrReplaceTempView('tmp_wikipedia_msg')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT geocoding.countryCode3, count(*) FROM tmp_wikipedia_msg
# MAGIC WHERE geocoding.countryCode3 is not null
# MAGIC GROUP BY geocoding.countryCode3
# MAGIC ORDER BY geocoding.countryCode3

# COMMAND ----------

# MAGIC %md ## 機械学習
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC * [Databricks Machine Learning ガイド](https://docs.microsoft.com/ja-jp/azure/databricks/applications/machine-learning/)
# MAGIC   - [Databricks AutoML](https://docs.microsoft.com/ja-jp/azure/databricks/applications/machine-learning/automl)
# MAGIC   - [モデルをトレーニング(手動)](https://docs.microsoft.com/ja-jp/azure/databricks/applications/machine-learning/train-model/)

# COMMAND ----------

# DBTITLE 1,ライブラリのimport
import os, warnings, sys, logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

# 評価値を返す関数を定義
def eval_metrics(actual, pred):
    rmse = np.sqrt( mean_squared_error(actual, pred) )
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# COMMAND ----------

# DBTITLE 1, 学習データの読み込み
np.random.seed(40)
csv_url = ('/dbfs/databricks-datasets/wine-quality/winequality-red.csv')

data = pd.read_csv(csv_url, sep=';')
train, test = train_test_split(data)

train_x = train.drop(['quality'], axis=1)
test_x = test.drop(['quality'], axis=1)

train_y = train[ ['quality'] ]
test_y = test[['quality']]

# COMMAND ----------

# DBTITLE 1,データ可視化・探索 (EDA)
# いろいろプロットを試してみてください
display(data)

# COMMAND ----------

# DBTITLE 1,モデルの構築(トレーニング)
import mlflow, mlflow.sklearn
mlflow.sklearn.autolog()

alpha = 0.02
l1_ratio = 0.01

with mlflow.start_run():
  lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
  lr.fit(train_x, train_y)

  pred = lr.predict(test_x)

  (rmse, mae, r2) = eval_metrics(test_y, pred)

  print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)

# COMMAND ----------

# DBTITLE 1,MLflowからモデルをロードする
## mlflow > Runページ > Aritifactcs > models にあるサンプルコードを参照してコードを書いてください

## 例: A. Run IDから **Spark Dataframe** のpython関数としてデプロイする
#
# import mlflow
# # 実際のRun IDで置き換えてください!
# logged_model = 'runs:/3b1aeb32524244b98006da4c7a2b7211/model'
# # モデルをロードする
# model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)
# model


## 例: B. Run IDから **Pandas Dataframe** のpython関数としてデプロイする
#
# import mlflow
# # 実際のRun IDで置き換えてください!
# logged_model = 'runs:/3b1aeb32524244b98006da4c7a2b7211/model'
# pd_model = mlflow.pyfunc.load_model(logged_model)
# pd_model


# 推定を実施する(スコアリングを実施する)対象のデータを読み込む

spark_df = (
  spark
  .read
  .format('csv')
  .option('Header', True)
  .option('inferSchema', True)
  .option('sep', ';')
  .load('/databricks-datasets/wine-quality/winequality-red.csv')
)

pandas_df = spark_df.toPandas()


## 推定する(モデルの適用)
# 
## A. Spark Dataframeの場合
# pred_df = spark_df.withColumn('pred', model(*spark_df.columns))
# display(pred_df)

## B. Pandas Dataframeの場合
# pred_array = pd_model.predict(pd_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## END
# MAGIC &copy; 2021 Copyright by Jixin Jia (Gin), Masahiko Kitamura
# MAGIC <hr>

# COMMAND ----------

# DBTITLE 1,Clean-up(このデモで作成したデータなどを削除する場合は以下を実行してください)
sql(f'DROP DATABASE IF EXISTS {dbname} CASCADE')
dbutils.fs.rm(delta_path, True)
dbutils.fs.rm(f'/tmp/{username}', True)
dbutils.fs.rm(homeDir, True)


# COMMAND ----------


