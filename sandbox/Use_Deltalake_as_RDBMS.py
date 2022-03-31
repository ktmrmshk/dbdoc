# Databricks notebook source
# MAGIC %md # Delta lakeをRDB(DWH)のように使う
# MAGIC 
# MAGIC 機械学習などでデータフレーム形式の中間データを保存する場合、RDBなどを使いたい場合などがあると思います。Databricks上では、Deltalakeがすぐに使えるため、これをRDBとして使うことができます。SQL(クエリ)処理はNotebookがアタッチしているクラスタ上のSparkで実施されるので、 DBサーバなどは不要です。sqliteのようにDeltalakeが使えます。

# COMMAND ----------

# MAGIC %md ## 1. Pandasの文脈で使用する
# MAGIC 
# MAGIC PandasのデータフレームをそのままDeltalakeに書き出す、読み込むことが可能です。

# COMMAND ----------

# サンプルのPandasデータフレームを用意しておく
import pandas as pd

p_df = pd.read_csv('https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv')
p_df

# COMMAND ----------

# MAGIC %md ### 1.1 Pandasデータフレーム全体をDeltaファイルフォーマットで保存する
# MAGIC 
# MAGIC Deltalakeの実態はファイルなので、CSVやJSONのように扱え、テーブルデータをそのまま保存できます。

# COMMAND ----------

# 書き出し(pandas => Delta)
(
  spark.createDataFrame(p_df)
  .write.format('delta').mode('append').save('dbfs:/tmp/your_dir/diamonds.delta')
)

# COMMAND ----------

#読み込み (Delta => pandas)
s_df = (
  spark.read.format('delta').load('dbfs:/tmp/your_dir/diamonds.delta')
  .toPandas()
)
s_df

# COMMAND ----------

# MAGIC %md ### 1.2 SQLを使って読み込む
# MAGIC 
# MAGIC DeltalakeはSQLでも操作できるので、SQLベースで読み込む。
# MAGIC 
# MAGIC SQLがそのままかけ、結果をPandasのデータフレームとして受けることができます。クエリ処理はDelta lakeで処理されるため高速です。
# MAGIC 
# MAGIC SQL内では、テーブル名として``delta.`Deltaファイルのパス`  ``で参照できます。ただ、無骨すぎるので、テーブル名を割り当てることもできます(後述)。

# COMMAND ----------

# SQLを使って読み出す
p_df = spark.sql('''
  SELECT * FROM delta.`dbfs:/tmp/your_dir/diamonds.delta`
  WHERE cut != "Fair"
  ORDER BY price DESC
  LIMIT 20 
''').toPandas()

p_df

# COMMAND ----------

# MAGIC %md ### 1.3 Deltaファイルとテーブル名を紐付ける
# MAGIC 
# MAGIC Deltaの実体はオブジェクトストレージ上のファイルです。一方で、SQLでは`データベース名.テーブル名`で参照します。テーブル名とDeltaファイルパスの紐付けをすることで通常のSQLのようにテーブル名でDeltaファイルを参照するできるようになります。
# MAGIC 
# MAGIC この「Deltaファイルパスとテーブル名の紐付け」はDatabricks上のHiveメタストア内で記録されています。Databricksのデータカタログからも参照可能になります。

# COMMAND ----------

spark.sql('''
  CREATE DATABASE IF NOT EXISTS testdb;
''')

spark.sql('''
  CREATE TABLE IF NOT EXISTS testdb.table_diamonds
  USING delta
  LOCATION 'dbfs:/tmp/your_dir/diamonds.delta';
''')

# COMMAND ----------

p_df = spark.sql('''
  SELECT * FROM testdb.table_diamonds
''').toPandas()

p_df

# COMMAND ----------

# MAGIC %md ## 2. RDBMSのようにSQLベースの文脈で使用する
# MAGIC 
# MAGIC 今まではpython/pandasからの利用でしたが、従来のRDBをSQLとして操作するのと同じ方法でDelta lakeを使用することもできます。
# MAGIC 
# MAGIC また、`spark.sql('好きなSQL').toPandas()`を使用することで、クエリ結果をpandasのデータフレームで受け取ることが可能です。

# COMMAND ----------

# MAGIC %sql
# MAGIC show databases;

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables from testdb;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM delta.`dbfs:/tmp/your_dir/diamonds.delta`
# MAGIC WHERE cut != "Fair"
# MAGIC ORDER BY price DESC
# MAGIC LIMIT 20 

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Databaseを作成する
# MAGIC CREATE DATABASE IF NOT EXISTS testdb_sql;
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS testdb_sql.table_foobar123
# MAGIC (id int, name string, age int, is_boosted boolean)
# MAGIC USING delta
# MAGIC LOCATION 'dbfs:/tmp/your_dir/table_foobar123.delta';

# COMMAND ----------

# MAGIC %sql
# MAGIC -- データをinsertしていく
# MAGIC INSERT INTO testdb_sql.table_foobar123 VALUES
# MAGIC (1, 'ABC', 123, true),
# MAGIC (2, 'FOO', 987, false),
# MAGIC (10, 'BAR', 555, true),
# MAGIC (101, 'HOGE', 142, true);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM testdb_sql.table_foobar123;

# COMMAND ----------

# MAGIC %sql -- DELETEを実施 (MERGE/UPSERTも通常のRDBMSと同じように実施できる)
# MAGIC DELETE FROM testdb_sql.table_foobar123
# MAGIC WHERE is_boosted == false;
# MAGIC 
# MAGIC SELECT * FROM testdb_sql.table_foobar123;

# COMMAND ----------

# MAGIC %sql -- テーブルの変更履歴
# MAGIC DESCRIBE HISTORY testdb_sql.table_foobar123;

# COMMAND ----------

# MAGIC %sql -- タイムトラベル(以前のテーブルの状態を参照する)
# MAGIC SELECT * FROM testdb_sql.table_foobar123 VERSION AS OF 2

# COMMAND ----------

# MAGIC %sql -- CLONE(データ実体を複製させずに、仮想的にテーブルをコピーする)
# MAGIC 
# MAGIC CREATE TABLE IF NOT EXISTS testdb_sql.clone_foobar123
# MAGIC CLONE testdb_sql.table_foobar123 VERSION AS OF 2;
# MAGIC 
# MAGIC SELECT * FROM testdb_sql.clone_foobar123;

# COMMAND ----------

# MAGIC %sql -- ユーザーからは独立したテーブルとして見える
# MAGIC show tables from testdb_sql;

# COMMAND ----------

# MAGIC %sql -- CLONEしたテーブルのタイムトラベルもオリジナルと独立している
# MAGIC describe history testdb_sql.clone_foobar123;

# COMMAND ----------

# MAGIC %sql -- CLONEテーブルの変更はオリジナルには関係なし
# MAGIC DELETE FROM testdb_sql.clone_foobar123 
# MAGIC WHERE name = 'ABC';
# MAGIC 
# MAGIC SELECT * FROM testdb_sql.clone_foobar123;

# COMMAND ----------

# MAGIC %sql -- オリジナルのテーブル
# MAGIC SELECT * FROM testdb_sql.table_foobar123

# COMMAND ----------

# MAGIC %md ## 3. Clean up
# MAGIC 
# MAGIC Delta lakeを削除するには、ファイルとテーブルメタデータの2つの削除する必要があります。

# COMMAND ----------

spark.sql('DROP TABLE IF EXISTS testdb_sql.table_foobar123;')
spark.sql('DROP TABLE IF EXISTS testdb_sql.clone_foobar123;')
spark.sql('DROP DATABASE IF EXISTS testdb_sql;')

dbutils.fs.rm('dbfs:/tmp/your_dir/diamonds.delta', True)

# COMMAND ----------


