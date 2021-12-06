# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC # EHRデータ分析
# MAGIC ## 1. ETL
# MAGIC 
# MAGIC <ol>
# MAGIC   <li> **データ**。マサチューセッツ州の約10,000人の患者について、**[synthea](https://github.com/synthetichealth/synthea)**を用いた患者のEHRデータの現実的なシミュレーションを使用しています</li>
# MAGIC   <li> **Ingestion and De-identification**: 我々は**pyspark**を使用してcsvファイルからデータを読み込み、患者のPIIを非識別化し、Delta Lakeに書き込んでいる</li>。
# MAGIC   <li> **データベースの作成**: その後、デルタテーブルを使用して、後続のデータ分析のために、患者の記録のデータベースを作成する</li>
# MAGIC </ol>
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC <img src="https://amir-hls.s3.us-east-2.amazonaws.com/public/rwe-uap.png" width=700>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. データをSparkのデータフレームに取り込む

# COMMAND ----------

# DBTITLE 1,ライブラリのインポートとダウンロードするcsvファイルの一覧表示
from pyspark.sql import functions as F, Window
ehr_path = '/databricks-datasets/rwe/ehr/csv'
display(dbutils.fs.ls(ehr_path)) ## display list of files

# COMMAND ----------

# DBTITLE 1,すべてのファイルをSparkデータフレームで取り込む
# create a python dictionary of dataframes
ehr_dfs = {}
for path,name in [(f.path,f.name) for f in dbutils.fs.ls(ehr_path) if f.name !='README.txt']:
  df_name = name.replace('.csv','')
  ehr_dfs[df_name] = spark.read.csv(path,header=True,inferSchema=True)

# Display number of records in each table
out_str="<h2>There are {} tables in this collection with:</h2><br>".format(len(ehr_dfs))
for k in ehr_dfs:
  out_str+='{}: <i style="color:Tomato;">{}</i> records <br>'.format(k.upper(),ehr_dfs[k].count())

displayHTML(out_str)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2. 患者のPIIを非識別化

# COMMAND ----------

# DBTITLE 1,暗号化関数の定義
from pyspark.sql.types import StringType, IntegerType, StructType, StructField
import hashlib

def encrypt_value(pii_col):
  sha_value = hashlib.sha1(pii_col.encode()).hexdigest()
  return sha_value

encrypt_value_udf = udf(encrypt_value, StringType())

# COMMAND ----------

# DBTITLE 1,PIIカラムに暗号化を適用
pii_cols=['SSN','DRIVERS','PASSPORT','PREFIX','FIRST','LAST','SUFFIX','MAIDEN','BIRTHPLACE','ADDRESS']
patients_obfuscated = ehr_dfs['patients']

for c in pii_cols:
  patients_obfuscated = patients_obfuscated.withColumn(c,F.coalesce(c,F.lit('null'))).withColumn(c,encrypt_value_udf(c))
display(patients_obfuscated)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. デルタレイクにテーブルを書き込む

# COMMAND ----------

## Specify the path to delta tables on dbfs
delta_root_path = "dbfs:/tmp/rwe-ehr/delta"

## to ensure fresh start we delete the path if it already exist
dbutils.fs.rm(delta_root_path, recurse=True)

## Create enounters table with renamed columns
(
  ehr_dfs['encounters']
  .withColumnRenamed('Id','Enc_Id')
  .withColumnRenamed('START', 'START_TIME')
  .withColumnRenamed('END', 'END_TIME')
  .write.format('delta').save(delta_root_path + '/encounters')
)

## Create providers table with renamed columns
(
  ehr_dfs['providers']
  .withColumnRenamed('NAME','Provider_Name')
  .withColumnRenamed('Id','PROVIDER')
  .write.format('delta').save(delta_root_path + '/providers')
)

## Create organizations table with renamed columns
(
  ehr_dfs['organizations']
  .withColumnRenamed('NAME','Org_Name')
  .withColumnRenamed('Id','ORGANIZATION')
  .withColumnRenamed('ADDRESS', 'PROVIDER_ADDRESS')
  .withColumnRenamed('CITY', 'PROVIDER_CITY')
  .withColumnRenamed('STATE', 'PROVIDER_STATE')
  .withColumnRenamed('ZIP', 'PROVIDER_ZIP')
  .withColumnRenamed('GENDER', 'PROVIDER_GENDER')
  .write.format('delta').save(delta_root_path + '/organizations')
)

## Create patients from dataframe with obfuscated PII
(
  patients_obfuscated
  .write.format('delta').save(delta_root_path + '/patients')
)

# COMMAND ----------

# DBTITLE 1,すべての患者の診察を含むテーブルを作成し、deltaに保存する
patients = spark.read.format("delta").load(delta_root_path + '/patients').withColumnRenamed('Id', 'PATIENT')
encounters = spark.read.format("delta").load(delta_root_path + '/encounters').withColumnRenamed('PROVIDER', 'ORGANIZATION')
organizations = spark.read.format("delta").load(delta_root_path + '/organizations')

(
  encounters
  .join(patients, ['PATIENT'])
  .join(organizations, ['ORGANIZATION'])
  .write.format('delta').save(delta_root_path + '/patient_encounters')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. データベースとテーブルの作成

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create Database
# MAGIC CREATE DATABASE IF NOT EXISTS rwd
# MAGIC     COMMENT "Database for real world data"
# MAGIC     LOCATION "dbfs:/tmp/rwe-ehr/databases";
# MAGIC 
# MAGIC DROP TABLE IF EXISTS rwd.encounters;
# MAGIC 
# MAGIC -- Create encounters table
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS rwd.encounters
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/tmp/rwe-ehr/delta/encounters';
# MAGIC 
# MAGIC -- Create providers table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS rwd.providers;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS rwd.providers
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/tmp/rwe-ehr/delta/providers';
# MAGIC 
# MAGIC -- Create organizations table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS rwd.organizations;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS rwd.organizations
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/tmp/rwe-ehr/delta/organizations';
# MAGIC 
# MAGIC -- Create patients table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS rwd.patients;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS rwd.patients
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/tmp/rwe-ehr/delta/patients';
# MAGIC 
# MAGIC -- Create patient encounter table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS rwd.patient_encounters;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS rwd.patient_encounters
# MAGIC USING DELTA
# MAGIC LOCATION 'dbfs:/tmp/rwe-ehr/delta/patient_encounters';

# COMMAND ----------

# MAGIC %sql SELECT * FROM rwd.patient_encounters

# COMMAND ----------

# MAGIC %md
# MAGIC Deltaの機能を使ってパフォーマンスの最適化ができるようになりました。詳しくは[ Delta Lake on Databricks ](https://docs.databricks.com/spark/latest/spark-sql/language-manual/optimize.html#optimize--delta-lake-on-databricks)をご覧ください。

# COMMAND ----------

# MAGIC %sql OPTIMIZE rwd.patients ZORDER BY (BIRTHDATE, ZIP, GENDER, RACE)

# COMMAND ----------

# MAGIC %sql OPTIMIZE rwd.patient_encounters ZORDER BY (REASONDESCRIPTION, START_TIME, ZIP, PATIENT)

# COMMAND ----------

# MAGIC %md
# MAGIC このETLノートブックをジョブとして設定し(https://docs.databricks.com/jobs.html#create-a-job)、所定のスケジュールに従って実行することができます。
# MAGIC 次に、データを素早く可視化するためのダッシュボードを作成します。次のノートブック(`./01-rwe-dashboard.R`)では、`R`でシンプルなダッシュボードを作成します。

# COMMAND ----------


