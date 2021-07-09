# Databricks notebook source
# MAGIC %md # 製造におけるIoTデータ分析 on Databricks
# MAGIC 
# MAGIC **注意**: 今回のデモ環境ではAzure Databricksを使用します。AWS, GCP上においても本デモと同等なDatabricks構成が可能です。
# MAGIC 
# MAGIC ## Part 1: Data エンジニアリング
# MAGIC このノートブックでは、Azure上でIoTのIngest、Processing、Analyticsを行うための以下のアーキテクチャをデモしています。デモでは以下のようなアーキテクチャが実装されています。
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_architecture.png" width=800>
# MAGIC 
# MAGIC ノートブックは以下のステップで進めていきます。:
# MAGIC 1. **Data Ingest** - Azure IoT Hubsからのリアルタイムの生のセンサーデータを、Azure StorageのDeltaフォーマットにストリーミングする。
# MAGIC 2. **Data Processing** - ストリームデータを生データテーブル(Bronze)から始めて、集約テーブル(Silver)を経て、サマリテーブル(Gold)を作成する。

# COMMAND ----------

# AzureML Workspace info (name, region, resource group and subscription ID) for model deployment
dbutils.widgets.text("Storage Account","<your ADLS Gen 2 account name>","Storage Account")

# COMMAND ----------

# MAGIC %md ## Step 1 - 環境のセットアップ
# MAGIC 
# MAGIC The pre-requisites are listed below:
# MAGIC 
# MAGIC 前提条件は下記の通りです。
# MAGIC 
# MAGIC ### 必要になるAzureサービス
# MAGIC * Azure IoT Hub 
# MAGIC * [Azure IoT Simulator](https://azure-samples.github.io/raspberry-pi-web-simulator/): 上記IoT Hubの設定、および、[this github repo](https://github.com/tomatoTomahto/azure_databricks_iot/blob/master/Manufacturing%20IoT/iot_simulator.js)で提供されるコード(センサーデータをIoT Hubに送信するプログラム)
# MAGIC * ADLS Gen2 Storageアカウント、および、`iot`という名前のコンテナ
# MAGIC 
# MAGIC ### Azure Databricksの設定
# MAGIC * 3ノード(最小)のDatabricks Cluster(Runtime: DBR 7.0以上)、かつ以下のライブラリ:
# MAGIC  * **Azure Event Hubs Connector for Databricks** - Maven coordinates `com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.17`
# MAGIC * 以下のSecrets(scope名:`iot`)
# MAGIC  * `iothub-cs` - IoT HubのConnection string **(重要 - [Event Hub Compatible](https://devblogs.microsoft.com/iotdev/understand-different-connection-strings-in-azure-iot-hub/) connection stringを使用すること)**
# MAGIC  * `adls_key` - ADLS storage accountへのアクセスキー **(重要 - [Access Key](https://raw.githubusercontent.com/tomatoTomahto/azure_databricks_iot/master/bricks.com/blog/2020/03/27/data-exfiltration-protection-with-azure-databricks.html))を使用すること**
# MAGIC * Notebookのウィジット:
# MAGIC  * `Storage Account` - ADLSのストレージアカウント

# COMMAND ----------

# 一時データ用のストレージアカウントへのアクセスを設定(Synapseへのプッシュ時に使用)
storage_account = dbutils.widgets.get("Storage Account")
spark.conf.set(f"fs.azure.account.key.{storage_account}.dfs.core.windows.net", dbutils.secrets.get("iot","adls_key"))

# ストレージ上のパスを設定する
ROOT_PATH = f"abfss://iot@{storage_account}.dfs.core.windows.net/manufacturing_demo/"
BRONZE_PATH = ROOT_PATH + "bronze/"
SILVER_PATH = ROOT_PATH + "silver/"
GOLD_PATH = ROOT_PATH + "gold/"
SYNAPSE_PATH = ROOT_PATH + "synapse/"
CHECKPOINT_PATH = ROOT_PATH + "checkpoints/"

# その他の初期化
IOT_CS = dbutils.secrets.get('iot','iothub-cs') # IoT Hub connection string (Event Hub Compatible)
ehConf = { 'eventhubs.connectionString':sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(IOT_CS) }

# Delta書き込み時の自動コンパクション、最適化の有効化
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled","true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled","true")

# Pythonライブラリのimport
import os, json, requests
from pyspark.sql import functions as F

# COMMAND ----------

# ストレージ上のクリーンアップ・残ファイルを削除(複数回実行時を考慮)
dbutils.fs.rm(ROOT_PATH, True)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- テーブルとViewのクリーンアップ
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_raw;
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_agg;
# MAGIC DROP TABLE IF EXISTS manufacturing.sensors_enriched;
# MAGIC DROP VIEW IF EXISTS manufacturing.facilities;
# MAGIC DROP VIEW IF EXISTS manufacturing.operating_limits;
# MAGIC DROP VIEW IF EXISTS manufacturing.parts_inventory;
# MAGIC DROP VIEW IF EXISTS manufacturing.ml_feature_view;
# MAGIC DROP TABLE IF EXISTS manufacturing.temperature_predictions;
# MAGIC DROP DATABASE IF EXISTS manufacturing;
# MAGIC 
# MAGIC -- データベースを作成(このデモで使用するテーブルは全てこのデータベース内に閉じる)
# MAGIC CREATE DATABASE IF NOT EXISTS manufacturing;

# COMMAND ----------

# MAGIC %md ## Step 2 - IoT Hubsからのデータのインジェスト
# MAGIC 
# MAGIC Azure Databricksは、IoTやイベントハブへのネイティブコネクタを提供しています。以下では、PySpark Structured Streamingを使用して、IoT Hubのデータストリームから読み取り、データを生のままDeltaに直接書き込む方法を紹介します。
# MAGIC 
# MAGIC 下図のように、IoT SimulatorがIoT Hubにペイロードを送信していることを確認します。
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/iot_simulator.gif" width=800>
# MAGIC 
# MAGIC 今回は以下のペイロードを持つデータ(1種類のみ)をIoT Hub上で扱います:
# MAGIC 1. **Sensor readings** - 次のペイロードを含む。 `date`,`timestamp`,`temperature`,`humidity`, `pressure`, `moisture`, `oxygen`, `radiation`, および `conductivity` フィールド。
# MAGIC 
# MAGIC IoT Hubからの生データストリームを、Azure Data Lake Storage上のDeltaテーブル形式に書き込みます。このBronzeテーブルには、データが流れ込んでくると即座にクエリを実行することができます。

# COMMAND ----------

# IoT Hubから受信するデータのスキーマ
schema = "facilityId string, timestamp timestamp, temperature double, humidity double, pressure double, moisture double, oxygen double, radiation double, conductivity double"

# EventHubs library for Databricksを使ってIoT Hubから直接読み込む
iot_stream = (
  spark.readStream.format("eventhubs")                                         # IoT HUBからの直接読み込み
    .options(**ehConf)                                                         # Event-Hub-enabled connect stringを使う
    .load()                                                                    # データをロードする
    .withColumn('reading', F.from_json(F.col('body').cast('string'), schema))  # メッセージから"body"ペイロードを抜き出す
    .select('reading.*', F.to_date('reading.timestamp').alias('date'))         # "date"フィールドを作成(Partitionとして使用)
)

# ストリームをDelta形式でADLS上に書き込む
write_iot_to_delta = ( iot_stream
    .select('date','facilityid','timestamp','temperature','humidity','pressure',
            'moisture','oxygen','radiation','conductivity')                    # 特定のフィールドを抽出する
    .writeStream.format('delta')                                               # Delta形式で書き出す
    .partitionBy('date')                                                       # 日付でPartitionする
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_raw")             # チェックポイントデータを書き出すディレクトリを指定
    .start(BRONZE_PATH + "sensors_raw")                                        # 書き出し処理を開始する
)

# データが入ってきてから外部テーブルを作成する
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_raw USING DELTA LOCATION "{BRONZE_PATH + "sensors_raw"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Deltaにデータが流れ始めると同時に、ストレージから直接データを照会することができます。
# MAGIC SELECT * FROM manufacturing.sensors_raw

# COMMAND ----------

# MAGIC %md ## Step 2 - Delta上でのData処理
# MAGIC 
# MAGIC 生のセンサーデータがAzure Storage上のBronze Deltaテーブルにストリーミングされている間に、このデータをSilverおよびGoldのデータセットに流すストリーミングパイプラインを作成することができます。
# MAGIC 
# MAGIC SilverとGoldのデータセットには以下のスキーマを使用します:
# MAGIC 
# MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/Manufacturing_dataflow.png" width=800>

# COMMAND ----------

# MAGIC %md ### 2a. Delta上でブロンズ(生データテーブル)からシルバー(集約テーブル)へ
# MAGIC 
# MAGIC 処理パイプラインの最初のステップでは、測定値をクリーニングし、5分間隔に集約します。
# MAGIC 
# MAGIC 時系列の値を集約しており、データが遅れて到着したり、データが変更されたりする可能性があるため、Deltaの[**MERGE**](https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/merge-into?toc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fazure-databricks%2Ftoc.json&bc=https%3A%2F%2Fdocs.microsoft.com%2Fen-us%2Fazure%2Fbread%2Ftoc.json)機能を使用して、レコードをターゲットテーブルにupsertします。
# MAGIC 
# MAGIC MERGEを使用すると、ソースレコードをターゲットストレージにアップサートすることができます。これは、時系列データを扱うときに便利です:
# MAGIC 
# MAGIC 1. データの到着が遅れることが多く、集計状態を更新する必要がある。
# MAGIC 2. ストリーミングデータがテーブルに入力されている間に、ヒストリカルデータをバックフィルする必要がある
# MAGIC 
# MAGIC ソースデータをストリーミングする際に、`foreachBatch()`を使って、データのマイクロバッチに対してマージを行うことができます。

# COMMAND ----------

# センサーデータを対象となるデルタテーブルにマージするための関数を作成
def merge_delta(incremental, target): 
  incremental.dropDuplicates(['date','window','facilityid']).createOrReplaceTempView("incremental")
  
  try:
    # 指定されたジョインキーを使ってレコードをターゲットテーブルにMERGEする
    incremental._jdf.sparkSession().sql(f"""
      MERGE INTO delta.`{target}` t
      USING incremental i
      ON i.date=t.date AND i.window = t.window AND i.facilityid = t.facilityid
      WHEN MATCHED THEN UPDATE SET *
      WHEN NOT MATCHED THEN INSERT *
    """)
  except:
    # ターゲットとなるテーブルが存在しない場合、テーブルを作成
    incremental.write.format("delta").partitionBy("date").save(target)
    
aggregate_sensors = (
  spark.readStream.format('delta').table("manufacturing.sensors_raw")          # ソースのDeltaテーブルからデータをストリームとして読み込む
    .groupBy('facilityid','date',F.window('timestamp','5 minutes'))            # 読み取り値を5分単位で集計
    .agg(F.avg('temperature').alias('temperature'),F.avg('humidity').alias('humidity'),F.avg('pressure').alias('pressure'),
         F.avg('moisture').alias('moisture'),F.avg('oxygen').alias('oxygen'),F.avg('radiation').alias('radiation'),F.avg('conductivity').alias('conductivity'))
    .writeStream                                                               # 結果のストリームを書き込む
    .foreachBatch(lambda i, b: merge_delta(i, SILVER_PATH + "sensors_agg"))    # Pass each micro-batch to a function
    .outputMode("update")                                                      # "update"モードでMergeを実施
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_agg")             # チェックポイントデータを書き出すディレクトリを指定
    .start()
)

# # Create the external tables once data starts to stream in
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_agg USING DELTA LOCATION "{SILVER_PATH + "sensors_agg"}"')
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql
# MAGIC -- データがリアルタイムで1時間ごとのテーブルにマージされると、すぐにクエリを実行できます。
# MAGIC SELECT * FROM manufacturing.sensors_agg WHERE facilityid = 'FAC-1' ORDER BY window ASC

# COMMAND ----------

# MAGIC %md ### 2b. Delta上でシルバーテーブル(集約テーブル)からゴールドテーブル(サマリテーブル)へ
# MAGIC 
# MAGIC 次に、データサイエンスやモデルのトレーニングに使用できる施設情報、容量、位置、在庫レベルなどのエンリッチメントデータに、センサーの読み取り値をストリーミング結合します。

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- エンリッチメントのために偽のデータでダミーのテーブルを作る
# MAGIC -- facilities：製造施設の地理的・容量的情報
# MAGIC CREATE OR REPLACE VIEW manufacturing.facilities AS
# MAGIC SELECT facilityid, 
# MAGIC   float(random()*20+30) as latitude, 
# MAGIC   float(-random()*40-80) as longitude,
# MAGIC   array('CA','OR','WA','TX','AZ','OH','AL','CO','FL','MN')[int(split(facilityid, '-')[1])] as state, 
# MAGIC   int(random()*1000+200) as capacity
# MAGIC FROM (SELECT DISTINCT facilityid FROM manufacturing.sensors_agg);
# MAGIC 
# MAGIC -- operating_limits：安全に動作させるための温度と圧力の境界線（境界線から外れると部品は故障する
# MAGIC CREATE OR REPLACE VIEW manufacturing.operating_limits AS
# MAGIC SELECT facilityid, 
# MAGIC     float(approx_percentile(temperature,0.10)) AS min_temp,
# MAGIC     float(approx_percentile(temperature,0.90)) AS max_temp,
# MAGIC     float(approx_percentile(pressure,0.10)) AS min_pressure,
# MAGIC     float(approx_percentile(pressure,0.90)) AS max_pressure
# MAGIC FROM manufacturing.sensors_agg
# MAGIC GROUP BY facilityid;
# MAGIC 
# MAGIC -- Parts Inventory：各施設における部品の日々の在庫量
# MAGIC CREATE OR REPLACE VIEW manufacturing.parts_inventory AS
# MAGIC SELECT facilityid, 
# MAGIC   date,
# MAGIC   float(random()*500+200) as inventory
# MAGIC FROM (SELECT DISTINCT facilityid, date FROM manufacturing.sensors_agg);

# COMMAND ----------

# Delta Silverテーブルからストリームを読み込み、共通のカラム（facilityid）で結合する
sensors_agg = spark.readStream.format('delta').option("ignoreChanges", True).table('manufacturing.sensors_agg')
sensors_enriched = (
  sensors_agg.join(spark.table('manufacturing.facilities'), 'facilityid')
    .join(spark.table('manufacturing.operating_limits'),'facilityid')
)

# 前述のようにMERGEを実行するforeachBatch関数にストリームを書き込みます。
merge_gold_stream = (
  sensors_enriched
    .withColumn('window',sensors_enriched.window.start)
    .writeStream 
    .foreachBatch(lambda i, b: merge_delta(i, GOLD_PATH + "sensors_enriched"))
    .option("checkpointLocation", CHECKPOINT_PATH + "sensors_enriched")         
    .start()
)

# データが入ってきてから外部テーブルを作成する
while True:
  try:
    spark.sql(f'CREATE TABLE IF NOT EXISTS manufacturing.sensors_enriched USING DELTA LOCATION "{GOLD_PATH + "sensors_enriched"}"')    
    break
  except:
    pass

# COMMAND ----------

# MAGIC %sql SELECT * FROM manufacturing.sensors_enriched WHERE facilityid="FAC-0" ORDER BY date, window

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Structureの最適化
# MAGIC 
# MAGIC デルタの`OPTIMIZE`コマンドは、一連のカラムに対してファイルのコンパクト化と多次元クラスタリング（ZORDERと呼ばれる）を実行します。これは、IoTデータでは通常、コンパクト化が必要な多くの小さなファイルが生成されるので便利です。また、ファシリティやタイムスタンプに基づいてテーブルを照会する際には、ファイルをスキップするためにこれらの列上でデータを順序付けすることでスピードアップできます。Deltaは[auto-optimize](https://docs.microsoft.com/en-us/azure/databricks/delta/optimizations/auto-optimize)を使って自動的にこの処理を行いますが、下記の[optimize](https://docs.microsoft.com/en-us/azure/databricks/delta/optimizations/file-mgmt)コマンドを使って定期的に行うこともできます。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 3つのテーブルをすべて最適化し、クエリとモデルトレーニングのパフォーマンスを向上させる
# MAGIC OPTIMIZE manufacturing.sensors_raw ZORDER BY facilityid, timestamp;
# MAGIC OPTIMIZE manufacturing.sensors_agg ZORDER BY facilityid, window;
# MAGIC OPTIMIZE manufacturing.sensors_enriched ZORDER BY facilityid, window;

# COMMAND ----------

# MAGIC %md ゴールドテーブルは、予測分析のための準備が整っています。これで、センサーの読み取り値が集約され、強化されました。次のステップは、コンポーネントの故障につながる特定の動作条件を予測することです（例：動作限界を超えた場合）。

# COMMAND ----------

# MAGIC %md
# MAGIC ####  時系列データにおけるDelta Lakeのメリット
# MAGIC A key component of this architecture is the Azure Data Lake Store (ADLS), which enables the write-once, access-often analytics pattern in Azure. However, Data Lakes alone do not solve challenges that come with time-series streaming data. The Delta storage format provides a layer of resiliency and performance on all data sources stored in ADLS. Specifically for time-series data, Delta provides the following advantages over other storage formats on ADLS:
# MAGIC 
# MAGIC このアーキテクチャの重要なコンポーネントは、Azure Data Lake Store（ADLS）であり、Azureにおけるwrite-once, access-oftenの分析パターンを実現しています。しかし、データレイクだけでは、時系列のストリーミングデータに伴う課題を解決することはできません。Deltaストレージフォーマットは、ADLSに保存されているすべてのデータソースに、回復力とパフォーマンスのレイヤーを提供します。特に時系列データの場合、DeltaはADLS上の他のストレージフォーマットに比べて以下のような利点があります。
# MAGIC 
# MAGIC |**要件**|**他の形式でADLSに保存**|**Delta形式でADLSに保存**|
# MAGIC |--------------------|-----------------------------|---------------------------|
# MAGIC |**バッチとストリーミングの統合**|データレイクは、CosmosDBのようなストリーミングストアと組み合わせて使われることが多く、複雑なアーキテクチャになっています。|ACIDに準拠したトランザクションにより、データエンジニアはADLSの同じ場所にストリーミングインジェストとヒストリカルバッチロードを行うことができる|
# MAGIC |**スキーマ強制・進化**|データレイクでは、スキーマが適用されないため、信頼性を高めるためにすべてのデータをリレーショナルデータベースにプッシュする必要があります。|スキーマはデフォルトで強化されています。データストリームに新しいIoTデバイスが追加されても、スキーマを安全に進化させることができるので、下流のアプリケーションが失敗することはない|
# MAGIC |**効率的なUpserts**|データレイクはインラインでのアップデートやマージをサポートしていないため、アップデートを行うためにはパーティション全体の削除や挿入が必要になります。|MERGEコマンドは、IoTの読み取りが遅れたり、リアルタイムエンリッチメントで使用したディメンションテーブルを変更したり、データを再処理する必要がある場合に有効です。|
# MAGIC |**ファイルのコンパクション**|時系列データをデータレイクにストリーミングすると、数百、数千の小さなファイルが生成されます。|Deltaのオートコンパクションがファイルサイズを最適化し、スループットと並列性を向上させる|
# MAGIC |**多次元のクラスタリング**|データレイクでは、パーティションのみのプッシュダウンフィルタリングが可能です。|タイムスタンプやセンサーIDなどのフィールドに時系列データをZORDERすることで、Databricksはこれらのカラムに対するフィルタリングや結合を、単純なパーティショニング技術に比べて最大100倍の速度で行うことができます。|

# COMMAND ----------

# MAGIC %md ## (補足) - SynapseやAzure Data Explorerにデータをストリームして配信する
# MAGIC 
# MAGIC The Data Lake is a great data store for historical analysis, data science, ML and ad-hoc visualization against *all hitorical* data using Databricks. However, a common use case in IoT projects is to serve an aggregated subset (3-6 months) or business level summary data to end users. Synapse SQL Pools provide *low latency, high concurrency* serving capabilities to BI tools for production-level reporting and BI. Follow the example notebook [here](https://databricks.com/notebooks/iiot/iiot-end-to-end-part-1.html) to stream our **GOLD** Delta table into a Synapse SQL Pool for reporting. 
# MAGIC 
# MAGIC データレイクは、履歴分析、データサイエンス、ML、Databricksを使用した*すべてのヒトリカル*データに対するアドホックなビジュアライゼーションに最適なデータストアです。しかし、IoTプロジェクトでの一般的なユースケースは、集約されたサブセット（3～6ヶ月）やビジネスレベルのサマリーデータをエンドユーザーに提供することです。Synapse SQL Poolsは、本番レベルのレポートやBIのために、BIツールに*低レイテンシー、高コンカレンシー*のサービング機能を提供します。こちら](https://databricks.com/notebooks/iiot/iiot-end-to-end-part-1.html)のノートの例に従って、レポート用にSynapse SQL Poolに**GOLD** Deltaテーブルをストリームします。
# MAGIC 
# MAGIC 
# MAGIC Similarily, another common use case in IoT projects is to serve real-time time-series reading into an operational dashboard used by operational engineers. Azure Data Explorer provides a real-time database for building operational dashboards on *current* data. Databricks can be used to stream either the raw or aggregated sensor data into ADX for operational serving. Follow the example snippet in the blog article [here](https://databricks.com/blog/2020/08/20/modern-industrial-iot-analytics-on-azure-part-3.html) to stream data from Delta into ADX. 
# MAGIC 
# MAGIC 
# MAGIC 同様に、IoTプロジェクトにおけるもう一つの一般的なユースケースは、運用エンジニアが使用する運用ダッシュボードにリアルタイムの時系列データを提供することです。Azure Data Explorerは、*現在の*データで運用ダッシュボードを構築するためのリアルタイムデータベースを提供します。Databricksは、生のセンサーデータまたは集約されたセンサーデータのいずれかをADXにストリーミングして、運用に供するために使用できます。ブログ記事[こちら](https://databricks.com/blog/2020/08/20/modern-industrial-iot-analytics-on-azure-part-3.html)のサンプルスニペットに従って、DeltaからADXにデータをストリームしてください。

# COMMAND ----------


