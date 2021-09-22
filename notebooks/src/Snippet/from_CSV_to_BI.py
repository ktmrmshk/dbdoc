# Databricks notebook source
# MAGIC %md ## 1. S3上のCSVファイルの確認

# COMMAND ----------

# MAGIC %fs ls s3://databricks-ktmr-s3/sample/airbnb/

# COMMAND ----------

# MAGIC %fs head s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ファイルの読み込み(一般的なデータレイク的な使い方)

# COMMAND ----------

# MAGIC %md ### 方法1: Pythonを使う(データフレーム)

# COMMAND ----------

schema='''
  id int,
  name string,
  host_id int,
  host_name string,
  neighbourhood_group string,
  neighbourhood string,
  latitude float,
  longitude float,
  room_type string,
  price int,
  minimum_nights int,
  number_of_reviews int,
  last_review date,
  reviews_per_month float,
  calculated_host_listings_count int,
  availability_365 int
'''


df = spark.read.csv('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv', header=True, schema=schema)

# データフレームの確認
display(df)
df.printSchema()

# COMMAND ----------

# 統計情報
display( df.summary() )

# COMMAND ----------

# MAGIC %md ### 方法2: SQLを使う(テーブル)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC -- 以前のデモ結果が残っている場合には削除する。
# MAGIC DROP DATABASE IF EXISTS csv_bi_demo123 CASCADE;
# MAGIC 
# MAGIC CREATE DATABASE csv_bi_demo123;
# MAGIC USE csv_bi_demo123;
# MAGIC 
# MAGIC CREATE TABLE airbnb_csv
# MAGIC (
# MAGIC   id int,
# MAGIC   name string,
# MAGIC   host_id int,
# MAGIC   host_name string,
# MAGIC   neighbourhood_group string,
# MAGIC   neighbourhood string,
# MAGIC   latitude float,
# MAGIC   longitude float,
# MAGIC   room_type string,
# MAGIC   price int,
# MAGIC   minimum_nights int,
# MAGIC   number_of_reviews int,
# MAGIC   last_review date,
# MAGIC   reviews_per_month float,
# MAGIC   calculated_host_listings_count int,
# MAGIC   availability_365 int
# MAGIC )
# MAGIC USING csv
# MAGIC OPTIONS (
# MAGIC   'Header' = 'True'
# MAGIC )
# MAGIC LOCATION 's3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM airbnb_csv where price > 100

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE EXTENDED airbnb_csv

# COMMAND ----------

# MAGIC %md ## 3. Delta Lake化
# MAGIC 
# MAGIC Deltaフォーマットで保存することにより、直接CSVファイルなどを読み込む場合に比べて **高速なクエリ処理** が実現できます。
# MAGIC 
# MAGIC また、高速性意外にも以下のような機能性も付与されます。
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

# COMMAND ----------

# Schemaを設定し、CSVファイルを読み込む(以前と同じコード。再掲)
schema='''
  id int,
  name string,
  host_id int,
  host_name string,
  neighbourhood_group string,
  neighbourhood string,
  latitude float,
  longitude float,
  room_type string,
  price int,
  minimum_nights int,
  number_of_reviews int,
  last_review date,
  reviews_per_month float,
  calculated_host_listings_count int,
  availability_365 int
'''

df = spark.read.csv('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv', header=True, schema=schema)


# Deltaフォーマットで保存する
df.write.format('delta').mode('overwrite').save('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.delta')

# COMMAND ----------

# DBTITLE 1,Deltaで書き出したS3上のファイルを確認
# MAGIC %fs ls s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.delta

# COMMAND ----------

# SQL/BIから参照できるようにTable化しておく
sql('''
  CREATE TABLE  csv_bi_demo123.airbnb_delta
  USING  delta
  LOCATION 's3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.delta' 
''')

# COMMAND ----------

# MAGIC %sql
# MAGIC -- SQLでテーブルが引けるかを確認
# MAGIC SELECT * FROM csv_bi_demo123.airbnb_delta 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 可視化1
# MAGIC SELECT * FROM csv_bi_demo123.airbnb_delta

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 可視化2
# MAGIC SELECT * FROM csv_bi_demo123.airbnb_delta

# COMMAND ----------


