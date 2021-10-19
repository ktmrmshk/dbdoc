# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <h1>Delta Lake機能プライマー</h1>
# MAGIC <table>
# MAGIC   <tr><th>作者(Mod)</th><th>Masahiko Kitamura</th></tr>
# MAGIC   <tr><th>作者(Original)</th><th>Jixin Jia (Gin)</th></tr>
# MAGIC   <tr><td>期日</td><td>2021/10/15</td></tr>
# MAGIC   <tr><td>バージョン</td><td>2.0</td></tr>
# MAGIC </table>
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

# COMMAND ----------

# DBTITLE 1,(デモ用準備) - ユニークなパス名を設定(ユーザー間の衝突回避)
import re

# Username を取得。
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Usernameをファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '_', username_raw).lower()
dbname = f'delta_db_{username}'

print(f'>>> username => {username}')
print(f'>>> dbname => {dbname}')

# COMMAND ----------

# MAGIC %md ##0. (参考)これまでのデータレイク

# COMMAND ----------

# DBTITLE 1,サンプルCSVデータ
# MAGIC %fs head /databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv

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
# MAGIC %fs head dbfs:/tmp/masahiko_kitamura_databricks_com/diamonds.json/part-00000-tid-1673204391485194256-d600e583-2f23-4605-82dc-ffbda2f18cdd-2003-1-c000.json

# COMMAND ----------

# DBTITLE 1,SQLも使えます
df.createOrReplaceTempView('diamonds')

df_sql = sql('''
  SELECT 
    carat, cut, color, clarity, price, x * y + z as magic_number
  FROM diamonds
''')

display( df_sql )

# COMMAND ----------

# MAGIC %md ### 従来のデータレイクの制限と限界
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

# COMMAND ----------

# MAGIC %md # 1. Delta Lakeの世界

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

# DBTITLE 1,(参考: DS向け) データのサマリを見る
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

# MAGIC %md
# MAGIC ## END
# MAGIC &copy; 2021 Copyright by Jixin Jia (Gin), Masahiko Kitamura
# MAGIC <hr>

# COMMAND ----------

# DBTITLE 1,Clean-up(このデモで作成したデータなどを削除する場合は以下を実行してください)
sql(f'DROP DATABASE {dbname} CASCADE')
dbutils.fs.rm(delta_path, True)
dbutils.fs.rm(homeDir, True)
dbutils.fs.rm(f'/tmp/{username}', True)




# COMMAND ----------


