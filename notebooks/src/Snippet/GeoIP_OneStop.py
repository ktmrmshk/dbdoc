# Databricks notebook source
# MAGIC %md # ユーザー設定
# MAGIC 
# MAGIC 適宜書き換えてください

# COMMAND ----------

# DBTITLE 0,ユーザー設定
#例) s3://foo-bucket/bar/　配下をデータ配置場所として使用する場合
S3_BUCKET_NAME       = 'foo-bucket'
S3_DIRECTORY         = 'bar'
GEOLITE2_LICENSE_KEY = 'GeoLite2のライセンスキーに置き換えてください'
DELTA_DATABASE_NAME  = 'test_geolite2'

# COMMAND ----------

# MAGIC %md # 使用方法
# MAGIC 
# MAGIC このNotebookを実行すると以下が実施されます。
# MAGIC 
# MAGIC 1. GeoLite2のCSVデータのダウンロード, S3への配置
# MAGIC 1. CSVからDeltaテーブルの作成
# MAGIC 
# MAGIC ### データ配置・テーブル名
# MAGIC |                |                    |
# MAGIC |----------------| ------------------ |
# MAGIC | CSV/Deltaデータ | 上記で設定したS3パス配下: `s3://{S3_BUCKET_NAME}/{S3_DIRECTORY}/`配下 |
# MAGIC | Deltaテーブル |   |
# MAGIC | AS番号テーブル - IPv4 | `{DELTA_DATABASE_NAME}`.`geolite2_asn_blocks_ipv4_delta` |
# MAGIC | AS番号テーブル - IPv6 | `{DELTA_DATABASE_NAME}`.`geolite2_asn_blocks_ipv6_delta` | 
# MAGIC | Countryテーブル - Location情報 |  `{DELTA_DATABASE_NAME}`.`geolite2_country_locations_en_delta` | 
# MAGIC | Countryテーブル - IP/LocationID - IPv4 |  `{DELTA_DATABASE_NAME}`.`geolite2_country_blocks_ipv4_delta` | 
# MAGIC | Countryテーブル - IP/LocationID - IPv6 |  `{DELTA_DATABASE_NAME}`.`geolite2_country_blocks_ipv6_delta` | 
# MAGIC | Cityテーブル - Location情報 |  `{DELTA_DATABASE_NAME}`.`geolite2_city_locations_en_delta` |
# MAGIC | Cityテーブル - IP/LocationID - IPv4 |  `{DELTA_DATABASE_NAME}`.`geolite2_city_blocks_ipv4_delta` |
# MAGIC | Cityテーブル - IP/LocationID - IPv6 |  `{DELTA_DATABASE_NAME}`.`geolite2_city_blocks_ipv6_delta` |

# COMMAND ----------

# MAGIC %md ##以降のセルは変更不要

# COMMAND ----------

#params
params={}
params['geolite2']={}
params['geolite2']['s3_bucket'] = S3_BUCKET_NAME
params['geolite2']['license_key'] = GEOLITE2_LICENSE_KEY
params['geolite2']['asn_csv_url'] = f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-ASN-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['city_csv_url'] =f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-City-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['country_csv_url'] = f"https://download.maxmind.com/app/geoip_download?edition_id=GeoLite2-Country-CSV&license_key={params['geolite2']['license_key']}&suffix=zip"
params['geolite2']['s3_base_dir'] = S3_DIRECTORY
params['geolite2']['asn_dirname']='ASN'
params['geolite2']['country_dirname']='Country'
params['geolite2']['city_dirname']='City'


# COMMAND ----------

# MAGIC %md # GeoLite2 CSVのダウンロード、S3配置

# COMMAND ----------

def upload_to_s3(local_filepath, bucket, upload_path):
  import boto3
  from botocore.exceptions import ClientError
  client = boto3.client('s3')
  try:
    ret=client.upload_file(local_filepath, bucket, upload_path)
  except ClientError as e:
    print(e)
    return False
  return True


def downlaod_file(url, local_filepath):
  import subprocess
  try:
    ret=subprocess.Popen(args=['curl', '-o', local_filepath, url],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE )
    print( ret.communicate() )
  except Exception as e:
    print(f'error=>{e}')
    

def get_and_upload_csv_to_s3(url, tagname, s3_bucket, upload_dir, tmpdir='/tmp'):
  '''
  url: geolite2 permalink
  tagname: fike name for temp use
  upload_dir: s3 directory name the file is uploded to
  '''
  import zipfile
  import os.path
  
  # get the files
  local_filepath = os.path.join(tmpdir, tagname)
  print(f'local_filepath => {local_filepath}')
  downlaod_file(url, local_filepath)
  
  # unzip
  z = zipfile.ZipFile(local_filepath)
  files = z.namelist()
  
  with z as f:
    f.extractall(tmpdir)
    
  # uploading to s3
  for n in files:
    if n.lower().endswith('.csv'):
      local=os.path.join(tmpdir, n)
      base=os.path.basename(n)
      s3_path=os.path.join(upload_dir, tagname, base)
      print(f'''uploading files to s3: {local} => s3://{params['geolite2']['s3_bucket']}/{s3_path}''')
      upload_to_s3(local, s3_bucket, s3_path)


# COMMAND ----------

# ASN
get_and_upload_csv_to_s3(
  params['geolite2']['asn_csv_url'],
  params['geolite2']['asn_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)

# Country
get_and_upload_csv_to_s3(
  params['geolite2']['country_csv_url'],
  params['geolite2']['country_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)


# City
get_and_upload_csv_to_s3(
  params['geolite2']['city_csv_url'],
  params['geolite2']['city_dirname'],
  params['geolite2']['s3_bucket'],
  params['geolite2']['s3_base_dir']
)

# COMMAND ----------

# MAGIC %md # GeoLite2のCSV => Deltaテーブル作成

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

  
  
s3_base = f'''s3a://{params['geolite2']['s3_bucket']}/{params['geolite2']['s3_base_dir']}'''

# ASN
createTableDelta(f'{s3_base}/ASN/GeoLite2-ASN-Blocks-IPv4.csv', 'geolite2_asn_blocks_ipv4_delta', DELTA_DATABASE_NAME)
createTableDelta(f'{s3_base}/ASN/GeoLite2-ASN-Blocks-IPv6.csv', 'geolite2_asn_blocks_ipv6_delta', DELTA_DATABASE_NAME)

# country
createTableDelta(f'{s3_base}/Country/GeoLite2-Country-Blocks-IPv4.csv', 'geolite2_country_blocks_ipv4_delta', DELTA_DATABASE_NAME)
createTableDelta(f'{s3_base}/Country/GeoLite2-Country-Blocks-IPv6.csv', 'geolite2_country_blocks_ipv6_delta', DELTA_DATABASE_NAME)
createTableDelta(f'{s3_base}/Country/GeoLite2-Country-Locations-en.csv', 'geolite2_country_locations_en_delta', DELTA_DATABASE_NAME)

# city
createTableDelta(f'{s3_base}/City/GeoLite2-City-Blocks-IPv4.csv', 'geolite2_city_blocks_ipv4_delta', DELTA_DATABASE_NAME)
createTableDelta(f'{s3_base}/City/GeoLite2-City-Blocks-IPv6.csv', 'geolite2_city_blocks_ipv6_delta', DELTA_DATABASE_NAME)
createTableDelta(f'{s3_base}/City/GeoLite2-City-Locations-en.csv', 'geolite2_city_locations_en_delta', DELTA_DATABASE_NAME)

# COMMAND ----------

# DBTITLE 1,(Optional) データ確認
sql(f'use {DELTA_DATABASE_NAME};')

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

# MAGIC %md # IPアドレスから地理情報を調べる(Lookup)

# COMMAND ----------

# DBTITLE 1,IP addressとCIDRブロックのマッピングで使うUDF
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

# DBTITLE 1,テスト用のランダムIPのDataFrameを準備
ips='''
78.111.216.254
218.108.247.146
28.111.231.220
38.122.132.153
32.103.10.172
207.134.12.84
61.167.199.90
57.131.76.227
31.87.240.21
159.215.114.209
111.5.31.50
98.48.65.209
168.157.24.255
42.161.173.238
23.213.73.177
233.23.214.180
102.100.40.45
246.60.187.154
71.209.81.8
115.250.101.47
'''

ipaddr = ips.splitlines()[1:]
df_ipaddr = spark.createDataFrame([{'ipaddr':ip} for ip in ipaddr])
df_ipaddr.createOrReplaceTempView('ipaddr_list')

display(sql(' select * from ipaddr_list'))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- [Optional]
# MAGIC -- range joinのため、bin=65536に設定しておく
# MAGIC SET spark.databricks.optimizer.rangeJoin.binSize=65536

# COMMAND ----------

# DBTITLE 1,1)  AS NumberのLook-up
# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM ipaddr_list ip
# MAGIC LEFT JOIN geolite2_asn_blocks_ipv4_delta asn
# MAGIC ON to_address_int(ip.ipaddr) BETWEEN to_network_address(asn.network) AND to_broadcast_address(asn.network)

# COMMAND ----------

# DBTITLE 1,2) CountryのLook-up
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

# DBTITLE 1,3) CityのLook-up
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

# DBTITLE 1,4) 全部入り (AS Number, Country, CityのLook-up)
# MAGIC %sql
# MAGIC -- (CityテーブルがCountryテーブルを内包している)
# MAGIC 
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
