# Databricks notebook source
# MAGIC %md # IPアドレスから地理情報を調べる(GeoLite2)
# MAGIC 
# MAGIC 
# MAGIC [MaxMind社](https://www.maxmind.com/en/home)がGeoLocationのデータベースGeoLite2を公開しています。CSVファイルでも公開されているので、これをDeltaテーブル化し、IP Addressから地理情報を取得する方法についてまとめてあります。
# MAGIC 
# MAGIC ユーザー登録すると、ダウンロードページからCSVファイルを取得できます。
# MAGIC ここでは、ダウンロードしたファイルを以下のS3上パスに配置した前提のコードになっています。
# MAGIC 
# MAGIC * ASN
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/ASN/GeoLite2-ASN-Blocks-IPv4.csv`
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/ASN/GeoLite2-ASN-Blocks-IPv6.csv`
# MAGIC * Country
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Blocks-IPv4.csv`
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Blocks-IPv6.csv`
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Locations-en.csv`
# MAGIC * City
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Blocks-IPv4.csv`
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Blocks-IPv6.csv`
# MAGIC   - `s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Locations-en.csv`

# COMMAND ----------

# MAGIC %md ## 1. GeoLite2のCSVをDeltaテーブル化する

# COMMAND ----------

# CSVからDeltaテーブルを作成
def createTableDelta(csv_path, table_name, dbname='db_geolite2'):
  delta_path = csv_path + '.delta'
  (
    spark
    .read
    .format('csv')
    .option('Header', True)
    .option('inferSchema', True)
    .load(csv_path)
    .write
    .format('delta')
    .mode('overwrite')
    .save(delta_path)
  )
  
  sql(f'''
    CREATE DATABASE IF NOT EXISTS {dbname}
  ''')
  
  sql(f'''
    CREATE TABLE IF NOT EXISTS {dbname}.{table_name}
    USING delta
    LOCATION '{delta_path}'
  ''')
  
# ASN
createTableDelta('s3://databricks-ktmr-s3/geolite2/ASN/GeoLite2-ASN-Blocks-IPv4.csv', 'geolite2_asn_blocks_ipv4_delta')
createTableDelta('s3://databricks-ktmr-s3/geolite2/ASN/GeoLite2-ASN-Blocks-IPv6.csv', 'geolite2_asn_blocks_ipv6_delta')

# country
createTableDelta('s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Blocks-IPv4.csv', 'geolite2_country_blocks_ipv4_delta')
createTableDelta('s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Blocks-IPv6.csv', 'geolite2_country_blocks_ipv6_delta')
createTableDelta('s3://databricks-ktmr-s3/geolite2/Country/GeoLite2-Country-Locations-en.csv', 'geolite2_country_locations_en_delta')

# city
createTableDelta('s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Blocks-IPv4.csv', 'geolite2_city_blocks_ipv4_delta')
createTableDelta('s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Blocks-IPv6.csv', 'geolite2_city_blocks_ipv6_delta')
createTableDelta('s3://databricks-ktmr-s3/geolite2/City/GeoLite2-City-Locations-en.csv', 'geolite2_city_locations_en_delta')

# COMMAND ----------

# MAGIC %md ## (Optional) データ確認

# COMMAND ----------

sql('use db_geolite2;')

print('===== AS Number Table ========')
print('---IPv4---: geolite2_asn_blocks_ipv4_delta')
display( sql(f'''select * from geolite2_asn_blocks_ipv4_delta limit 5''') )

print('---IPv6---: geolite2_asn_blocks_ipv6_delta')
display( sql(f'''select * from geolite2_asn_blocks_ipv6_delta limit 5''') )

print('')
print('======= Country Table =======')
print('--- CIDR => geoname_id ---')
print('---IPv4---: geolite2_country_blocks_ipv4_delta')
display( sql(f'''select * from geolite2_country_blocks_ipv4_delta limit 5''') )

print('---IPv6---: geolite2_country_blocks_ipv6_delta')
display( sql(f'''select * from geolite2_country_blocks_ipv6_delta limit 5''') )

print('--- geoname_id => country name ----: geolite2_country_locations_en_delta')
display( sql(f'''select * from geolite2_country_locations_en_delta limit 5''') )

print('')
print('======= City Table =======')
print('--- CIDR => geoname_id ---')
print('---IPv4---: geolite2_city_blocks_ipv4_delta')
display( sql(f'''select * from geolite2_city_blocks_ipv4_delta limit 5''') )

print('---IPv6---: geolite2_city_blocks_ipv6_delta')
display( sql(f'''select * from geolite2_city_blocks_ipv6_delta limit 5''') )

print('--- geoname_id => country name ----: geolite2_country_locations_en_delta')
display( sql(f'''select * from geolite2_city_locations_en_delta limit 5''') )

# COMMAND ----------

# MAGIC %md ## 2. IP addressとCIDRブロックのマッピングで使うUDF

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

# MAGIC %md ## 3. Look upのサンプルコード(SQL)

# COMMAND ----------

# MAGIC %md ### 3.0 サンプルデータ作成

# COMMAND ----------

# sample data
ipaddr_list = [{'ipaddr':'165.225.111.47'}, {'ipaddr':'8.8.8.8'}, {'ipaddr':'1.1.1.1'}]
df_ipaddr = spark.createDataFrame(ipaddr_list)
df_ipaddr.createOrReplaceTempView('ipaddr_list')

display(sql(' select * from ipaddr_list'))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- [Optional]
# MAGIC -- range joinのため、bin=65536に設定しておく
# MAGIC SET spark.databricks.optimizer.rangeJoin.binSize=65536

# COMMAND ----------

# MAGIC %md ### 3.1 AS NumberのLook-up

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM ipaddr_list ip
# MAGIC LEFT JOIN geolite2_asn_blocks_ipv4_delta asn
# MAGIC ON to_address_int(ip.ipaddr) BETWEEN to_network_address(asn.network) AND to_broadcast_address(asn.network)

# COMMAND ----------

# MAGIC %md ### 3.2 CountryのLook-up

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   * 
# MAGIC FROM (
# MAGIC   SELECT *
# MAGIC   FROM ipaddr_list ip
# MAGIC   LEFT JOIN geolite2_country_blocks_ipv4_delta c
# MAGIC   ON to_address_int(ip.ipaddr) BETWEEN to_network_address(c.network) AND to_broadcast_address(c.network) ) fact
# MAGIC LEFT JOIN geolite2_country_locations_en_delta loc
# MAGIC ON fact.geoname_id = loc.geoname_id

# COMMAND ----------

# MAGIC %md ### 3.3 CityのLook-up

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   * 
# MAGIC FROM (
# MAGIC   SELECT *
# MAGIC   FROM ipaddr_list ip
# MAGIC   LEFT JOIN geolite2_city_blocks_ipv4_delta c
# MAGIC   ON to_address_int(ip.ipaddr) BETWEEN to_network_address(c.network) AND to_broadcast_address(c.network) ) fact
# MAGIC LEFT JOIN geolite2_city_locations_en_delta loc
# MAGIC ON fact.geoname_id = loc.geoname_id

# COMMAND ----------

# MAGIC %md ### 3.4 全部入り (AS Number, Country, CityのLook-up)
# MAGIC 
# MAGIC (CityテーブルがCountryテーブルを内包している)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ret_asn.ipaddr,
# MAGIC   ret_asn.network as asn_network,
# MAGIC   ret_asn.autonomous_system_number,
# MAGIC   ret_asn.autonomous_system_organization,
# MAGIC   ret_city.network as city_network,
# MAGIC   --ret_city.geoname_id,
# MAGIC   ret_city.registered_country_geoname_id,
# MAGIC   ret_city.represented_country_geoname_id,
# MAGIC   ret_city.is_anonymous_proxy,
# MAGIC   ret_city.is_satellite_provider,
# MAGIC   ret_city.postal_code,
# MAGIC   ret_city.latitude,
# MAGIC   ret_city.longitude,
# MAGIC   ret_city.accuracy_radius,
# MAGIC   ret_city.locale_code,
# MAGIC   ret_city.continent_code,
# MAGIC   ret_city.continent_name,
# MAGIC   ret_city.country_iso_code,
# MAGIC   ret_city.country_name,
# MAGIC   ret_city.subdivision_1_iso_code,
# MAGIC   ret_city.subdivision_1_name,
# MAGIC   ret_city.subdivision_2_iso_code,
# MAGIC   ret_city.subdivision_2_name,
# MAGIC   ret_city.city_name,
# MAGIC   ret_city.metro_code,
# MAGIC   ret_city.time_zone,
# MAGIC   ret_city.is_in_european_union
# MAGIC FROM
# MAGIC   (
# MAGIC     SELECT
# MAGIC       *
# MAGIC     FROM
# MAGIC       ipaddr_list ip
# MAGIC       LEFT JOIN geolite2_asn_blocks_ipv4_delta asn ON to_address_int(ip.ipaddr) BETWEEN to_network_address(asn.network)
# MAGIC       AND to_broadcast_address(asn.network)
# MAGIC   ) AS ret_asn
# MAGIC   LEFT JOIN (
# MAGIC     SELECT
# MAGIC       *
# MAGIC     FROM
# MAGIC       (
# MAGIC         SELECT
# MAGIC           *
# MAGIC         FROM
# MAGIC           ipaddr_list ip
# MAGIC           LEFT JOIN geolite2_city_blocks_ipv4_delta c ON to_address_int(ip.ipaddr) BETWEEN to_network_address(c.network)
# MAGIC           AND to_broadcast_address(c.network)
# MAGIC       ) fact
# MAGIC       LEFT JOIN geolite2_city_locations_en_delta loc ON fact.geoname_id = loc.geoname_id
# MAGIC   ) AS ret_city ON ret_asn.ipaddr = ret_city.ipaddr
