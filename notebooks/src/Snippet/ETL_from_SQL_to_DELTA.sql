-- Databricks notebook source
-- MAGIC %md
-- MAGIC 
-- MAGIC # データソースへのアクセス
-- MAGIC 
-- MAGIC Databricks上でAzure Data Lake Storage(ADLS)と連携する場合、以下の3通りの方法があります。
-- MAGIC 
-- MAGIC 1. ADLSのアクセスキーを用いてアクセスする
-- MAGIC 1. ADDのサービスプリンシパルを用いてアクセスする
-- MAGIC 1. SAS(トークン)を用いてアクセスする
-- MAGIC 
-- MAGIC ここでは、ADLSのアクセスキーを用いてアクセスする方法をみていきます。アクセスキーがnotebook上に平文で書かれるため、取り扱いに注意してください。
-- MAGIC (本番環境ではSecret機能を使ってアクセスキーを管理することが推奨されます)

-- COMMAND ----------

-- DBTITLE 0,ストレージへ直接アクセス
-- MAGIC %python
-- MAGIC 
-- MAGIC # ストレージアカウント、コンテナ、アクセスキーの設定
-- MAGIC storage_account = 'your_ADLS_storage_account'
-- MAGIC storage_container = 'your_ADLS_storage_container'
-- MAGIC access_key = 'Your-ADLS-Container-Access-Key'
-- MAGIC 
-- MAGIC # spark環境変数に上記の内容を反映
-- MAGIC spark.conf.set( f'fs.azure.account.key.{storage_account}.dfs.core.windows.net', access_key)
-- MAGIC 
-- MAGIC # File Path Base
-- MAGIC print( f'File Path Base=>"abfss://{storage_container}@{storage_account}.dfs.core.windows.net/" \n')
-- MAGIC 
-- MAGIC # ストレージにアクセスできるか確認 (list)
-- MAGIC dbutils.fs.ls(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/')

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # SQL ServerのデータをSQLでロード・Deltaテーブル化する

-- COMMAND ----------

-- DBTITLE 1,SQL Server上のDatabase/Tableに接続する
--　SQLサーバに接続する (temp viewとして"my_temp_view123"を定義。後のSelect文で使用する)
CREATE OR REPLACE TEMP VIEW my_temp_view123
USING org.apache.spark.sql.jdbc
OPTIONS (
  url "jdbc:sqlserver://<SQLサーバのホスト名>.database.windows.net:<ポート番号>;database=<SQLサーバのDatabase名>",
  dbtable "<SQLサーバにある参照するテーブル名>",
  user "<SQLサーバアクセスのためのユーザー名>",
  password "<SQLサーバアクセスのためのパスワード>"
);

-- COMMAND ----------

--- Viewの内容の確認
SELECT * FROM my_temp_view123

-- COMMAND ----------

--- Schemaの確認
DESCRIBE my_temp_view123

-- COMMAND ----------

-- DBTITLE 1,そのまま(ETLなしで)DELTA Tableに書き出す
-- 注意: SQLでは変数が使用できないため、<storage_container>および<storage_account>を適宜置換した後、実行してください。

DROP TABLE IF EXISTS my_delta_table;

CREATE TABLE my_delta_table
USING DELTA
PARTITIONED BY ( <パーティションするカラム名1>, <パーティションするカラム名2>.....  )
LOCATION "abfss://<storage_container>@<storage_account>.dfs.core.windows.net/databricks-sandbox/my_delta_table"
AS (
  SELECT *
  FROM my_temp_view123
)



-- COMMAND ----------

-- DBTITLE 1,ストレージに書き出されたDeltaファイルを確認する
-- MAGIC %fs ls abfss://<storage_container>@<storage_account>.dfs.core.windows.net/databricks-sandbox/my_delta_table

-- COMMAND ----------

-- DBTITLE 1,上記のDelta Tableからデータを加工して、さらに新しいDelta Table(サマリテーブル)を作成する
CREATE TABLE my_delta_summary
USING DELTA
PARTITIONED BY ( <パーティションするカラム名1>, <パーティションするカラム名2>.....  )
LOCATION "abfss://<storage_container>@<storage_account>.dfs.core.windows.net/databricks-sandbox/my_delta_summary"
AS (
  SELECT <どこかのカラム名>, count(*)
  FROM my_delta_table
  GROUP BY <どこかのカラム名>
  ORDER BY <どこかのカラム名> desc
)
