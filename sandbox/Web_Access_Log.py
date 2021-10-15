# Databricks notebook source
# MAGIC %md
# MAGIC # Webアクセスログ解析
# MAGIC 
# MAGIC このNotebookではクラウドのオブジェクトストレージ上(AWS S3, Azure Blob Storage, GCSなど)にあるWebアクセスログのデータ分析を実施します。
# MAGIC データ処理の流れは以下の通りです。
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC 
# MAGIC 1. オブジェクトストレージ上からログファイルを読み込み、Deltaテーブル化します(生データログ)。
# MAGIC 1. 上記で取り込んだデータのみを使って、アドホック分析を実施します。
# MAGIC 1. 続いて、地理的な分析要素と組み合わせるため、GeoLocation(GeoLite2)データをDeltaテーブル化します(生データIP-Geo)。
# MAGIC 1. これら2つのテーブルから分析・可視化のためにJOINしたVIEWを作成します(アクセスデータ+地理データ)。
# MAGIC 1. 地理情報を含めた可視化をします。
# MAGIC 1. 上記で実施したグラフをダッシュボード化します。
# MAGIC 1. 機械学習へ続きます。
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/webaccess_overview.jpg" width="900">

# COMMAND ----------

# MAGIC %md 
# MAGIC ## ログデータの読み込み / Delta Lake化

# COMMAND ----------

# DBTITLE 1,rawデータの読み込み
raw_df = spark.read.text('s3://databricks-ktmr-s3/var/log/access.log.*.gz')
display(raw_df)

# COMMAND ----------

# DBTITLE 1,レコード数の確認
raw_df.count()

# COMMAND ----------

# DBTITLE 1,rawデータからparse・整形(bronzeテーブル)
from pyspark.sql.functions import split, regexp_extract, col, to_timestamp

split_df = (
    raw_df.select(
      regexp_extract('value', r'^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', 1).alias('src_ip'),
      regexp_extract('value', r'\[(.+?)\]', 1).alias('time_string'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 1).alias('method'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 2).alias('path'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ "(.+) (.+) (HTTP.+?)"', 3).alias('version'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+)', 1).cast('int').alias('status_code'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+)', 2).cast('int').alias('content_size'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+) "(.+?)" "(.+?)" "(.+?)"', 3).alias('host2'),
      regexp_extract('value', r'^\S+ \S+ \S+ \S+ \S+ ".+HTTP.+?" (\d+) (\d+) "(.+?)" "(.+?)" "(.+?)"', 4).alias('user_agent')
    )
  .withColumn( 'timestamp', to_timestamp(  col('time_string'), 'dd/MMM/yyyy:HH:mm:ss Z') )
  .drop('time_string')
  .filter( col('timestamp').isNotNull() ) 
)

display(split_df)

# COMMAND ----------

# DBTITLE 1,Delta形式で保存
split_df.write.format('delta').mode('overwrite').save('s3://databricks-ktmr-s3/var/delta/access_log.delta')

# COMMAND ----------

# DBTITLE 1,テーブルを登録(Deltaテーブル)
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS ktmrdb.access_log
# MAGIC USING delta
# MAGIC LOCATION 's3://databricks-ktmr-s3/var/delta/access_log.delta'

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQLでログ解析

# COMMAND ----------

# DBTITLE 1,パス別アクセス数(テーブル)
# MAGIC %sql
# MAGIC SELECT path, count(1) as cnt FROM ktmrdb.access_log
# MAGIC WHERE method = 'GET'
# MAGIC GROUP BY path
# MAGIC ORDER BY cnt desc
# MAGIC limit 100

# COMMAND ----------

# DBTITLE 1,パス別アクセス数(グラフ)
# MAGIC %sql
# MAGIC SELECT path, count(1) as cnt FROM ktmrdb.access_log
# MAGIC WHERE method = 'GET'
# MAGIC GROUP BY path
# MAGIC ORDER BY cnt desc
# MAGIC limit 100

# COMMAND ----------

# DBTITLE 1,ステータスコード(分布)
# MAGIC %sql
# MAGIC SELECT status_code, count(1) as cnt FROM ktmrdb.access_log
# MAGIC GROUP BY status_code
# MAGIC ORDER BY cnt desc

# COMMAND ----------

# DBTITLE 1,ステータスコード(時系列)
# MAGIC %sql
# MAGIC WITH access_log_with_ts AS (
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   date_trunc('MONTH', timestamp) as month,
# MAGIC   date_trunc('HOUR', timestamp) as hour,
# MAGIC   date_trunc('MINUTE', timestamp) as minute
# MAGIC FROM ktmrdb.access_log
# MAGIC -- limit 10
# MAGIC )
# MAGIC SELECT hour, status_code, count(1) as count_access FROM access_log_with_ts
# MAGIC GROUP BY hour, status_code
# MAGIC ORDER BY hour

# COMMAND ----------

# DBTITLE 1,日次のアクセス数
# MAGIC %sql
# MAGIC 
# MAGIC WITH access_log_with_ts AS (
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   date_trunc('DAY', timestamp) as day,
# MAGIC   date_trunc('HOUR', timestamp) as hour,
# MAGIC   date_trunc('MINUTE', timestamp) as minute
# MAGIC FROM ktmrdb.access_log
# MAGIC -- limit 10
# MAGIC )
# MAGIC SELECT day, status_code, count(1) as count_access FROM access_log_with_ts
# MAGIC GROUP BY day, status_code
# MAGIC ORDER BY day

# COMMAND ----------

# DBTITLE 1,ユニークユーザごとのアクセス数
# MAGIC %sql
# MAGIC 
# MAGIC SELECT src_ip, user_agent, count(1) as cnt FROM ktmrdb.access_log
# MAGIC GROUP BY src_ip, user_agent
# MAGIC ORDER BY cnt desc
# MAGIC limit 100

# COMMAND ----------

# DBTITLE 1,ユニークユーザー(時系列)
# MAGIC %sql
# MAGIC WITH access_log_with_ts AS (
# MAGIC SELECT 
# MAGIC   *,
# MAGIC   date_trunc('DAY', timestamp) as day,
# MAGIC   date_trunc('HOUR', timestamp) as hour,
# MAGIC   date_trunc('MINUTE', timestamp) as minute
# MAGIC FROM ktmrdb.access_log
# MAGIC -- limit 10
# MAGIC )
# MAGIC SELECT day, count(1) as cnt
# MAGIC FROM (
# MAGIC   SELECT src_ip, user_agent, day FROM access_log_with_ts
# MAGIC   GROUP BY src_ip, user_agent, day
# MAGIC )
# MAGIC GROUP BY day
# MAGIC ORDER BY day

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pythonでも分析可能です

# COMMAND ----------

from pyspark.sql.functions import col, to_date, date_format, hour, date_trunc


df = (
  spark
  .read.format('delta').load('s3://databricks-ktmr-s3/var/delta/access_log.delta')
  .withColumn( 'month',  date_trunc( 'MONTH' , col('timestamp') ) )
  .withColumn( 'date',   date_trunc( 'DATE'  , col('timestamp') ) )
  .withColumn( 'hour',   date_trunc( 'HOUR'  , col('timestamp') ) )
  .withColumn( 'minute', date_trunc( 'MINUTE', col('timestamp') ) )
).cache()

display(df)

# COMMAND ----------

display(
  df.groupBy('minute').count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## GeoIPテーブルの作成
# MAGIC 
# MAGIC すでに別のNotebookで作成したGeoIPテーブル`db_geolite2.geolite2_city_blocks_ipv4_delta`を使用して、そこからアクセスログに含まれるIPのLocationテーブルを作成します。

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM db_geolite2.geolite2_city_blocks_ipv4_delta where represented_country_geoname_id is not null limit 10

# COMMAND ----------

# IP addressとCIDRブロックのマッピングで使うUDF
import pandas as pd
from pyspark.sql.functions import pandas_udf, col

@pandas_udf('long')
def to_network_address(cidr: pd.Series) -> pd.DataFrame:
  import ipaddress as ip
  return cidr.apply(lambda x: int(ip.IPv4Network(x).network_address) )

spark.udf.register('to_network_address', to_network_address)


@pandas_udf('long')
def to_broadcast_address(cidr: pd.Series) -> pd.DataFrame:
  import ipaddress as ip
  return cidr.apply(lambda x: int(ip.IPv4Network(x).broadcast_address) )

spark.udf.register('to_broadcast_address', to_broadcast_address)


@pandas_udf('long')
def to_address_int(cidr: pd.Series) -> pd.DataFrame:
  import ipaddress as ip
  return cidr.apply(lambda x: int(ip.IPv4Address(x)) )

spark.udf.register('to_address_int', to_address_int)

# COMMAND ----------

# DBTITLE 1,アクセスログに含まれるIPを抽出
# MAGIC %sql
# MAGIC create or replace temp view ipaddr_list
# MAGIC as select distinct src_ip as ipaddr from ktmrdb.access_log where src_ip != ''

# COMMAND ----------

# DBTITLE 1,アクセスログに含まれるIPのGeoLocationのテーブルを作成
sql('SET spark.databricks.optimizer.rangeJoin.binSize=65536;')
sql('USE db_geolite2')

geoip_df = sql('''
SELECT
  ret_asn.ipaddr,
  ret_asn.network as asn_network,
  ret_asn.autonomous_system_number,
  ret_asn.autonomous_system_organization,
  ret_city.network as city_network,
  --ret_city.geoname_id,
  ret_city.registered_country_geoname_id,
  ret_city.represented_country_geoname_id,
  ret_city.is_anonymous_proxy,
  ret_city.is_satellite_provider,
  ret_city.postal_code,
  ret_city.latitude,
  ret_city.longitude,
  ret_city.accuracy_radius,
  ret_city.locale_code,
  ret_city.continent_code,
  ret_city.continent_name,
  ret_city.country_iso_code,
  ret_city.country_name,
  ret_city.subdivision_1_iso_code,
  ret_city.subdivision_1_name,
  ret_city.subdivision_2_iso_code,
  ret_city.subdivision_2_name,
  ret_city.city_name,
  ret_city.metro_code,
  ret_city.time_zone,
  ret_city.is_in_european_union
FROM
  (
    SELECT
      *
    FROM
      ipaddr_list ip
      LEFT JOIN geolite2_asn_blocks_ipv4_delta asn ON to_address_int(ip.ipaddr) BETWEEN to_network_address(asn.network)
      AND to_broadcast_address(asn.network)
  ) AS ret_asn
  LEFT JOIN (
    SELECT
      *
    FROM
      (
        SELECT
          *
        FROM
          ipaddr_list ip
          LEFT JOIN geolite2_city_blocks_ipv4_delta c ON to_address_int(ip.ipaddr) BETWEEN to_network_address(c.network)
          AND to_broadcast_address(c.network)
      ) fact
      LEFT JOIN geolite2_city_locations_en_delta loc ON fact.geoname_id = loc.geoname_id
  ) AS ret_city ON ret_asn.ipaddr = ret_city.ipaddr
''')

geoip_df.write.format('delta').mode('overwrite').save('s3://databricks-ktmr-s3/var/delta/geoip.delta')
sql("CREATE TABLE IF NOT EXISTS ktmrdb.geoip USING delta LOCATION 's3://databricks-ktmr-s3/var/delta/geoip.delta'")

# COMMAND ----------

# DBTITLE 1,作成したテーブルの確認
# MAGIC %sql
# MAGIC SELECT * FROM ktmrdb.geoip

# COMMAND ----------

# DBTITLE 1,IP(ipaddr)から高速にlookupするためのindexing
# MAGIC %sql
# MAGIC OPTIMIZE ktmrdb.geoip ZORDER BY (ipaddr)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 地理情報の利用(テーブル結合)

# COMMAND ----------

# DBTITLE 1,国コード 変換用のテーブルを用意
# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS ktmrdb.country_code
# MAGIC (
# MAGIC   Country string,
# MAGIC   2letter_code string,
# MAGIC   3letter_code string
# MAGIC )
# MAGIC USING CSV
# MAGIC LOCATION 's3://databricks-ktmr-s3/sample/countries.csv'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ktmrdb.country_code

# COMMAND ----------

# DBTITLE 1,地理情報分析・可視化のためのVIEWを作成
# MAGIC %sql
# MAGIC CREATE VIEW IF NOT EXISTS ktmrdb.access_location
# MAGIC AS (
# MAGIC WITH access_ranking AS (
# MAGIC   SELECT src_ip, count(1) as cnt
# MAGIC   FROM ktmrdb.access_log
# MAGIC   GROUP BY src_ip
# MAGIC )
# MAGIC SELECT * FROM access_ranking a
# MAGIC JOIN ktmrdb.geoip g
# MAGIC ON a.src_ip = g.ipaddr
# MAGIC JOIN  ktmrdb.country_code c
# MAGIC ON g.country_iso_code = c.2letter_code
# MAGIC ORDER BY a.cnt desc
# MAGIC 
# MAGIC )

# COMMAND ----------

# DBTITLE 1,国別アクセス数
# MAGIC %sql
# MAGIC SELECT 3letter_code as country, sum(cnt) as sum_cnt FROM ktmrdb.access_location
# MAGIC GROUP BY 3letter_code
# MAGIC ORDER BY sum_cnt desc
# MAGIC limit 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## ダッシュボードの作成
# MAGIC 
# MAGIC Notebook上のプロット結果をまとめてダッシュボードを作成することができます。
# MAGIC これにより、通常のモニタ業務向けのUIとして使用できます。
# MAGIC 
# MAGIC <img src="https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/notebook_dashboard.jpg" witdh="1200">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 機械学習へ直結
# MAGIC 
# MAGIC このNotebookで作成したテーブルデータやアドホックに作成したVIEWのデータを使って、シームレスに機械学習が実施できます。
# MAGIC 機械学習のサンプル(コードを含む)は以下を参照ください。
# MAGIC 
# MAGIC * [Databricks ソリューションアクセラレータ](https://databricks.com/jp/solutions/accelerators)

# COMMAND ----------


