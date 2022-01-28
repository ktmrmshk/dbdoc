# Databricks notebook source
# MAGIC %md # Spark Structured Streaming

# COMMAND ----------

# MAGIC %md ## 特徴
# MAGIC 
# MAGIC * **Sparkのバッチ処理とほぼ同じAPIでストリーミング処理を構成できる(バッチとストリーミングの統合)**
# MAGIC   * ストリーミングもDataframeとして扱う
# MAGIC   * 行の終わりのが確定していないテーブル
# MAGIC   
# MAGIC   <img src="https://spark.apache.org/docs/latest/img/structured-streaming-stream-as-a-table.png" width=600/>
# MAGIC 
# MAGIC   
# MAGIC * **マイクロバッチで処理する**
# MAGIC   * 新しいレコード(差分レコード)のみが入力され、バッチ処理対象になる。
# MAGIC   * 出力にも差分レコードの処理結果が出力される。
# MAGIC   * 処理内容によって出力モード`append`, `update`, `complete`の3種類を使い分ける必要がある
# MAGIC 
# MAGIC   <img src="https://files.training.databricks.com/images/streaming-timeline-1-sec.png" width=800>
# MAGIC 
# MAGIC * **時系列データの取り扱い機能の充実**
# MAGIC   * 遅れて届いたデータの扱い(Watermarking)
# MAGIC   * 時間Windowでの集計(移動平均など)
# MAGIC 
# MAGIC * **Sparkの特徴であるスケーラビリティ、対障害性、冪等性がそのまま享受できる。**
# MAGIC * **Delta Lakeとの高い親和性(データソース、シンクとしての利用)**

# COMMAND ----------

# MAGIC %md ## Quick Start
# MAGIC 
# MAGIC まずは、コードを見てみましょう。
# MAGIC 
# MAGIC 以下のコードはSparkのStructured Streamingを使って、ストリーミングソースからデータを読み込み(受信)、ETL処理を施し、ストリーミングシンクにデータを書き出す(送信)コードです。
# MAGIC 
# MAGIC ```python
# MAGIC 
# MAGIC #　1. 読み込み
# MAGIC df1 = (
# MAGIC   spark
# MAGIC   .readStream
# MAGIC   .format( 'json' )
# MAGIC   .schema( data_schema )
# MAGIC   .option( 'maxFilesPerTrigger', 1 )
# MAGIC   .load('/databricks-datasets/definitive-guide/data/activity-data/*.json')
# MAGIC )
# MAGIC 
# MAGIC # 2. ETL処理
# MAGIC df2 = (
# MAGIC   df1
# MAGIC   .filter(df1.x > 0.0)
# MAGIC   .select('Arrival_Time', 'x')
# MAGIC   .withColumn('Arrival_ts', to_timestamp( df1.Arrival_Time) )
# MAGIC )
# MAGIC 
# MAGIC # 3. 書き出し
# MAGIC ret = (
# MAGIC   df2
# MAGIC   .writeStream
# MAGIC   .format('delta')
# MAGIC   .option('path', '/data/sensor_data.delta')
# MAGIC   .option('checkpointLocation', '/data/sensor_data.checkpoint')
# MAGIC   .outputMode('append')
# MAGIC   .trigger(processingTime="3 seconds")
# MAGIC   .start()
# MAGIC )
# MAGIC ```
# MAGIC 
# MAGIC ### 1. 読み込み
# MAGIC 
# MAGIC * `spark.readStream`によって、ストリーミング形式(Structured Streaming)でDataframeを生成される。
# MAGIC 
# MAGIC * バッチ処理とは異なり、読み込むデータのスキーマ推定(inferSchema)が不可のため、スキーマはユーザーが指定する必要がある。
# MAGIC   * ただし、Autoloader(`.format('cloudFiles')`)を使う場合、スキーマ推定、スキーマ進化が使用可能。
# MAGIC 
# MAGIC * ストリーミングのソースとして以下がサポートされる。[詳細はこちら](https://docs.databricks.com/spark/latest/structured-streaming/data-sources.html)
# MAGIC   * Apache Kafka
# MAGIC   * Aamazon Kinesis
# MAGIC   * Azure EventHub
# MAGIC   * Files(オブジェクトストレージ上のファイル, CSV, JSON, parquet, Avroなど): Autoloader
# MAGIC   * Delta Lake
# MAGIC 
# MAGIC * Databricks上で、スリーミングソースとしてファイルを使う場合、`Autoloader`の使用が推奨
# MAGIC   * ディレクトリに新しく追加されたファイルを認識して読み込む(ユーザー側で読み込み済みファイルの管理が不要)
# MAGIC   * SQSなどとの連携も可能で、効率的に新規ファイルをリストすることが可能
# MAGIC   * スキーマ推定、スキーマ進化が使用可能
# MAGIC 
# MAGIC * オプションで一回のマイクロバッチで読み込みサイズを制御するなどが可能。データソースによって指定可能なオプションは異なるので、[ドキュメント](https://docs.gcp.databricks.com/spark/latest/structured-streaming/data-sources.html)を参照。
# MAGIC 
# MAGIC ### 2. ETL/データ加工
# MAGIC 
# MAGIC * Sparkのバッチ処理とほぼ同様に処理が書ける。
# MAGIC 
# MAGIC * ストリーミングでは意味のない処理はサポートされていない。例えばSort系の処理など。
# MAGIC 
# MAGIC * マイクロバッチのため、今まで受信したデータ全てが使用できない。処理が以下の2種類に分類される。
# MAGIC   * マイクロバッチ内で閉じる処理(Stateless): フィルター、Join、数値変換処理など
# MAGIC   * マイクロバッチ内で閉じない処理(Stateful): count, sum, maxなどの集約を伴う処理
# MAGIC 
# MAGIC * Steatefulな処理の場合、Spark内部で状態データを保持する。
# MAGIC * SQLでも記載可能
# MAGIC 
# MAGIC 
# MAGIC ### 3. 書き出し
# MAGIC 
# MAGIC * チェックポイントのファイルパス(`checkpointLocation`)を指定する。
# MAGIC   * データソースから読み込んだデータのOffset管理など
# MAGIC 
# MAGIC * 以下のデータシンクがサポートされいる。[ドキュメント]()
# MAGIC   * Apache Kafka
# MAGIC   * Files
# MAGIC   * Delta Lake
# MAGIC   * Console: デバッグ用
# MAGIC   * Memory: デバッグ用
# MAGIC   * 上記以外の任意のデータシンク: `foreach()`, `foreachBatch()`でカスタムで書き込み処理を実装可能
# MAGIC 
# MAGIC 
# MAGIC * 2.での処理内容によってoutputされるモード(OutputMode)が分かれる。
# MAGIC   * ** Output Modes **
# MAGIC 
# MAGIC | Mode   | Example | Notes |
# MAGIC | ------------- | ----------- |
# MAGIC | **Complete** (全体) | `.outputMode("complete")` | 更新されたResult Table全体がシンクに書き込まれる。テーブル全体の書き込みをどのように処理するかは、個々のシンクの実装が決定する。 |
# MAGIC | **Append** (追記) | `.outputMode("append")`     | 最後のトリガー以降にResult Tableに追加された新しい行のみがシンクに書き込まれます。 |
# MAGIC | **Update** (更新部分のみ)| `.outputMode("update")`     | 結果テーブルのうち、最後のトリガー以降に更新された行のみがシンクに出力されます。 |
# MAGIC 
# MAGIC <img src="https://spark.apache.org/docs/latest/img/structured-streaming-example-model.png">
# MAGIC 
# MAGIC * マイクロバッチのトリガータイミング(Trigger)は以下の通り。
# MAGIC   * **Trigger**
# MAGIC   
# MAGIC | Trigger Type                           | Example | Notes |
# MAGIC |----------------------------------------|-----------|-------------|
# MAGIC | 指定なし                            |  | デフォルト - システムが前のクエリーの処理を完了すると同時に、クエリーが実行されます。 |
# MAGIC | 固定のインターバル           | `.trigger(processingTime='10 seconds')` | クエリーはマイクロバッチで実行され、ユーザーが指定した間隔でキックオフされます |
# MAGIC | 1度のみ(one-time)  | `.trigger(once=True)` | クエリは1つのマイクロバッチを実行し、利用可能なすべてのデータを処理した後、自動的に停止します。 |
# MAGIC | Continuous w/fixed checkpoint interval | `.trigger(continuous='1 second')` | クエリーは低レイテンシーで連続的に処理するモードで実行されます。2.3.2 の _EXPERIMENTAL_ です。 |

# COMMAND ----------

# MAGIC %md ## デモの構成(全体像)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/wholeimage.png" width=1000>
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * データソース: Kafkaなどのmessage queue、もしくは、オブジェクトストレージに随時アップロードされてくるファイル
# MAGIC * データソースからストリーミングとしてデータを受信し、Deltaテーブル化
# MAGIC * Delta Lake内でクレンジング、サマリテーブル化を実施(ストリーミング差分更新)

# COMMAND ----------

# DBTITLE 1,環境設定 - ファイルパス・テーブル名の設定
# 以下にユニーク値を設定してください
username = "ユニークな名前で置き換えてください！"
# 例: 
username = "kitamura1112"

base_path = f'/tmp/{username}/streaming_lab'
db_name = f'db_{username}'

dbutils.fs.rm(base_path, True)
spark.sql(f''' DROP DATABASE IF EXISTS `{db_name}` CASCADE; ''')
spark.sql(f''' CREATE DATABASE `{db_name}`; ''')
spark.sql(f''' use {db_name}; 
''')

print(f'''
  * base_path: {base_path} 
  * database: {db_name}
''')

# COMMAND ----------

# MAGIC %md ## Streamingのデータソース
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/data_sources.png" width=250>
# MAGIC 
# MAGIC チュートリアルで使用するデータ: `/databricks-datasets/definitive-guide/data/activity-data/*.json`
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC * **注意** ここでは、デモのため、すでに全てのファイルがアップロードされた状態から始めています。ただし、本来であれば、ファイルをストリーミングのデータソースにする場合、オブジェクトストレージ上に随時ファイルがアップロードされる状況になることが多いと思います。

# COMMAND ----------

# DBTITLE 1,ファイル一覧
# MAGIC %fs ls /databricks-datasets/definitive-guide/data/activity-data/

# COMMAND ----------

# DBTITLE 1,JSONファイルの中身を確認
# MAGIC %fs head /databricks-datasets/definitive-guide/data/activity-data/part-00018-tid-730451297822678341-1dda7027-2071-4d73-a0e2-7fb6a91e1d1f-0-c000.json

# COMMAND ----------

# スキーマの定義(StreamingはinferSchemaしている時間がないので、ユーザーがスキーマを与える必要がある)
data_schema='''
Arrival_Time long,
Creation_Time long,
Device string,
Index long,
Model string,
User string,
gt string,
x double,
y double,
z double
'''

# COMMAND ----------

# MAGIC %md ## 1. データソースからStreamingを受信する (Rawテーブル)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/raw_table.png" width=500>
# MAGIC 
# MAGIC -----
# MAGIC 
# MAGIC * **注意** ファイルをストリーミングのソースとして読み込む場合、`Autoloader`の使用が推奨です。ただし、ここではデモのため(徐々にファイルを読み込むために)、従来のファイルソースのStreamingを使っていきます。

# COMMAND ----------

# ファイルパスなどの指定
raw_delta_path    = base_path + '/raw.delta'
raw_chkpoint_path = base_path + '/raw.checkpoint'
raw_table_name = 'table_raw'
print(f'''
  * raw_delta_path: {raw_delta_path}
  * raw_chkpoint_path: {raw_chkpoint_path}
  * raw_table_name: {raw_table_name}
''')

# 読み込み側 (従来のファイルソースのStreaming)
# 徐々に読み込むデモを実施するため、このチュートリアルではこちらを使用する。
df_raw = (
  spark
  .readStream
  .format( 'json' )
  .schema( data_schema )
  .option( 'maxFilesPerTrigger', 1 )
  .load('/databricks-datasets/definitive-guide/data/activity-data/*.json')
)

# 読み込み側 (autoloader, スキーマエボリューションやSQS連携など、従来のファイルソースで読み込むより機能性が高い)
# df_raw = (
#   spark
#   .readStream
#   .format( 'cloudFiles' )
#   .option( 'cloudFiles.format', 'json')
#   .schema( data_schema )
#   .load('/databricks-datasets/definitive-guide/data/activity-data/*.json')
# )

# COMMAND ----------

# 書き込み側
ret_raw = (
  df_raw
  .writeStream
  .format('delta')
  .option('checkpointLocation', raw_chkpoint_path)
  .option('path', raw_delta_path)
  .trigger(processingTime='3 seconds')
  .outputMode('append')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {raw_table_name}
  USING delta
  LOCATION '{raw_delta_path}';
''')

# COMMAND ----------

display(
  spark.sql(f'''
    SELECT * FROM {raw_table_name}
  ''')
)

# COMMAND ----------

# MAGIC %md ## 2. データのクレンジング(シルバーテーブル)
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/silver_table.png" width=700>

# COMMAND ----------

# DBTITLE 1,User属性テーブルを事前に用意(Dimentionテーブル、静的)
import json

user_dimension=json.loads('''
[
{ "user_id": "a", "state": "東京"},
{ "user_id": "b", "state": "千葉"},
{ "user_id": "c", "state": "神奈川"},
{ "user_id": "d", "state": "埼玉"},
{ "user_id": "e", "state": "栃木"},
{ "user_id": "f", "state": "群馬"},
{ "user_id": "g", "state": "長野"},
{ "user_id": "h", "state": "茨城"},
{ "user_id": "i", "state": "山梨"}
]
''')

df_user = spark.createDataFrame(user_dimension)
display(df_user)

# COMMAND ----------

# ファイルパスなどの指定
cleaned_delta_path    = base_path + '/cleaned.delta'
cleaned_chkpoint_path = base_path + '/cleaned.checkpoint'
cleaned_table_name = 'table_cleaned'
print(f'''
  * cleaned_delta_path: {cleaned_delta_path}
  * cleaned_chkpoint_path: {cleaned_chkpoint_path}
  * cleaned_table_name: {cleaned_table_name}
''')


# 読み込み側
df_raw = (
  spark
  .readStream
  .format( 'delta' )
  .load(raw_delta_path) # rawテーブル(delta)のパスから読み込む
)

# COMMAND ----------

# DBTITLE 1,SQLを使った場合
# クレンジング処理 + Join処理
df_raw.createOrReplaceTempView('st_table_raw')

df_cleaned = sql('''
SELECT 
  *,
  to_timestamp(Arrival_Time / 1000.0) as Arrival_ts
FROM 
  st_table_raw
''')

# 書き込み
ret_cleaned = (
  df_cleaned
  .writeStream
  .format('delta')
  .option('checkpointLocation', cleaned_chkpoint_path)
  .option('path', cleaned_delta_path)
  .outputMode('append')
  .start()
)

# COMMAND ----------

# クレンジング処理 + Join処理
df_cleaned = (
  df_raw
  .withColumn('Arrival_ts', to_timestamp( col('Arrival_Time') / 1000.0 )  )
  .withColumn('Creation_ts', to_timestamp( col('Creation_Time') / 1000000000.0 )  )

  .select('Arrival_ts', 'Creation_ts', 'Device', 'Index', 'Model', 'User', 'gt', 'x', 'y', 'z')
  .join(df_user, df_raw.User == df_user.user_id, "left_outer" )
)

# 書き込み
ret_cleaned = (
  df_cleaned
  .writeStream
  .format('delta')
  .option('checkpointLocation', cleaned_chkpoint_path)
  .option('path', cleaned_delta_path)
  .outputMode('append')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {cleaned_table_name}
  USING delta
  LOCATION '{cleaned_delta_path}';
''')

# COMMAND ----------

# DBTITLE 1,テーブルの確認(SQL編)
# MAGIC %sql 
# MAGIC -- テーブルの確認(SQL編)
# MAGIC 
# MAGIC SELECT * FROM table_cleaned

# COMMAND ----------

# DBTITLE 1,テーブルの確認(Python編)
df_tmp = spark.read.format('delta').load(cleaned_delta_path)

display( df_tmp )

# COMMAND ----------

# MAGIC %md ## 3. ビジネスサマリテーブル
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/gold_table.png" width=1000>

# COMMAND ----------

# ファイルパスなどの指定
summary_delta_path    = base_path + '/summary.delta'
summary_chkpoint_path = base_path + '/summary.checkpoint'
summary_table_name = 'table_summary'
print(f'''
  * summary_delta_path: {summary_delta_path}
  * summary_chkpoint_path: {summary_chkpoint_path}
  * summary_table_name: {summary_table_name}
''')


# 読み込み側
df_cleaned = (
  spark
  .readStream
  .format( 'delta' )
  .load(cleaned_delta_path) # cleanedテーブル(delta)のパスから読み込む
)

# COMMAND ----------

# 集約・統計
df_summary = (
  df_cleaned
  .groupBy('User', 'Device', 'state')
  .agg( 
    count('*').alias('count'),
    sum('x').alias('sum_x'), 
    mean('y').alias('mean_y'),
    stddev('z').alias('stddev_z'),
    max('Index').alias('max_index'), 
    percentile_approx('z', 0.5).alias('median_z')
  )
)



# COMMAND ----------

# 書き出し
ret_summary = (
  df_summary
  .writeStream
  .format('delta')
  .option('checkpointLocation', summary_chkpoint_path)
  .option('path', summary_delta_path)
  .outputMode('complete')
  .start()
)

# COMMAND ----------

# tableの登録
spark.sql(f'''
  CREATE TABLE IF NOT EXISTS {summary_table_name}
  USING delta
  LOCATION '{summary_delta_path}';
''')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM table_summary;

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY table_summary;

# COMMAND ----------

# MAGIC %md ## 環境のクリーンアップ

# COMMAND ----------

dbutils.fs.rm(base_path, True)
spark.sql(f''' DROP DATABASE IF EXISTS `{db_name}` CASCADE; ''')

# COMMAND ----------

# MAGIC %md # 応用編

# COMMAND ----------

# MAGIC %md ## Auto Loader(`cloudFiles`)の利用
# MAGIC 
# MAGIC **ユースケース**: オブジェクトストレージ上に逐次アップロードされるファイルを効率的に読み込む
# MAGIC 
# MAGIC 補足: 基礎編においてはファイルの取り込みにおいて、簡単のため、Spark従来の`files`ソースを使用しました。実際のワークロードでは、機能性が高いAuto Loaderを使用することをおすすめします。Auto Loaderを使うことで、ファイルアップロードをSQS通知で効率的に読み込む機能、効率的なファイルリスト、スキーマ推定・進化などの付加機能が利用できます。
# MAGIC 
# MAGIC ドキュメント: Auto Loader [日本語(一部)](https://qiita.com/taka_yayoi/items/df143647dcf5942b51c6) | [オリジナル(full)](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html) 
# MAGIC 
# MAGIC サンプルコードは、基礎編のファイル読み込みのパートでコメントアウトされておりますので、参照ください。
# MAGIC 
# MAGIC (再掲)
# MAGIC ```
# MAGIC # 読み込み側 (autoloader, スキーマエボリューションやSQS連携など、従来のファイルソースで読み込むより機能性が高い)
# MAGIC df_raw = (
# MAGIC   spark
# MAGIC   .readStream
# MAGIC    .format( 'cloudFiles' )
# MAGIC 　 .option( 'cloudFiles.format', 'json')
# MAGIC 　 .schema( data_schema )
# MAGIC    .load('/databricks-datasets/definitive-guide/data/activity-data/*.json')
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## 新規ファイルのみ対象にバッチ処理を実施する(Trigger Once)
# MAGIC 
# MAGIC **ユースケース:** オブジェクトストレージ上に随時アップロードされてくるCSVファイルを1時間に一度Delta Lakeに取り込む
# MAGIC 
# MAGIC `writeStream`において、`.trigger(once=True)`を指定する。また、Streaming処理は非同期(non-blocking)になるため、Trigger Once処理が終わるまでコードを止める場合は、`.awaitTermination()`を追加する。
# MAGIC 
# MAGIC ```
# MAGIC # 3. 書き出し
# MAGIC ret = (
# MAGIC   df2
# MAGIC   .writeStream
# MAGIC   .format('delta')
# MAGIC   .option('path', '/data/sensor_data.delta')
# MAGIC   .option('checkpointLocation', '/data/sensor_data.checkpoint')
# MAGIC   .outputMode('append')
# MAGIC   .trigger(once=True)
# MAGIC   .start()
# MAGIC   .awaitTermination()
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Watermarkとwindow関数
# MAGIC 
# MAGIC ストリーミングで扱うデータは時系列データであることが多く、5分毎の平均、10分ウィンドウの5分間隔のスライディングウィンドウでの集計などが求められます。
# MAGIC これらは、イベント時間(現場でログが記録された時刻)をベースにした処理になります。例えば、ネットワーク環境などでログデータのDatabricksへの到着が大きく遅延する状況などが考られます。また、この時、`Append`処理であればレコードの順序が入れ替わるだけ済みますが、`Aggregation`を伴う処理の場合はイベント時間を考慮した処理が必要になります。
# MAGIC 
# MAGIC Spark Structured Streamingではこれらの処理をシンプルに実現する機能が提供されています。
# MAGIC 
# MAGIC * [Handling Late Data and Watermarking](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html#handling-late-data-and-watermarking)

# COMMAND ----------

# MAGIC %md ## Kafkaなどの連携
# MAGIC 
# MAGIC Databricks上では、Structured Streamingのデータソースとして、Apache Kafka, AWS Kinesis, Azure EventHubなどとの連携が可能です。
# MAGIC 
# MAGIC ドキュメント: 
# MAGIC  * Amazon Kinesis | Databricks on AWS: [日本語](https://qiita.com/taka_yayoi/items/6987fe95dfaf463aba06) | [オリジナル](https://docs.databricks.com/spark/latest/structured-streaming/kinesis.html)
# MAGIC  * [Apache Kafka](https://docs.databricks.com/spark/latest/structured-streaming/kafka.html)

# COMMAND ----------

# MAGIC %md ### Apache Kafkaをデータソースにする例
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

# MAGIC %md ## Delta Live Tables(DLT)との使い分け
# MAGIC 
# MAGIC Delta Live Tablesは、Structured Streamingを基盤として、その上で以下のような付加機能を提供するフレームワークになります。
# MAGIC 
# MAGIC * パイプラインのユーザーインターフェース
# MAGIC * テーブル単位でのリネージ
# MAGIC * パイプラインのデータ品質管理・エラーハンドリング拡張
# MAGIC * 宣言型ETLパイプライン
# MAGIC * シンプルなデプロイメント機能
# MAGIC 
# MAGIC 詳しくは、ドキュメントを参照ください。
# MAGIC 
# MAGIC * [Delta Live Tablesユーザーガイド](https://qiita.com/taka_yayoi/items/6726ad1edfa92d5cd0e9)
# MAGIC * [Delta Live Tablesでインテリジェントデータパイプラインを実装する5つのステップ](https://qiita.com/taka_yayoi/items/8466ceb6b689541327fc)

# COMMAND ----------

# MAGIC %md # 参考
# MAGIC 
# MAGIC * [Structured Streaming](https://docs.databricks.com/spark/latest/structured-streaming/index.html)

# COMMAND ----------


