-- Databricks notebook source
-- MAGIC %md
-- MAGIC 
-- MAGIC # ADLSへのアクセス
-- MAGIC 
-- MAGIC Databricks上でAzure Data Lake Storage(ADLS)と連携する場合、以下の3通りの方法があります。
-- MAGIC 
-- MAGIC 1. [ADLSのアクセスキーを用いてアクセスする](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/azure/adls-gen2/azure-datalake-gen2-get-started)
-- MAGIC 1. [ADDのサービスプリンシパルを用いてアクセスする](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/azure/adls-gen2/azure-datalake-gen2-sp-access)
-- MAGIC 1. [SAS(トークン)を用いてアクセスする](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/azure/adls-gen2/azure-datalake-gen2-sas-access)
-- MAGIC 
-- MAGIC ここでは、ADLSのアクセスキーを用いてアクセスする方法をみていきます。
-- MAGIC まずはじめに、理解のため、アクセスキーを平文で扱う例を見ていきます。
-- MAGIC その後、Secret機能を用いて、アクセスキーが隠蔽した形(redacted)で扱う方法を見ていきます。
-- MAGIC 
-- MAGIC 前半のコードはアクセスキーがnotebook上に平文で書かれるため、取り扱いに注意してください。
-- MAGIC (本番環境ではSecret機能を使ってアクセスキーを管理することが推奨されます)

-- COMMAND ----------

-- MAGIC %md ## アクセスキーを平文のまま使うコード

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

-- DBTITLE 1,DBFSコマンドでファイルを一覧を取得する
-- MAGIC %fs ls abfss://<storage_container>@<storage_account>.dfs.core.windows.net/

-- COMMAND ----------

-- MAGIC %md ## Serect機能を用いて、アクセスキーを隠蔽する
-- MAGIC 
-- MAGIC Secretは、Azure Key Vaultと連携して、機密情報を隠蔽したままコード上で扱うことを可能にする機能です。
-- MAGIC 
-- MAGIC Secretsの登録方法は以下のサイトの
-- MAGIC * ストレージ アカウントと BLOB コンテナーを作成する
-- MAGIC * Azure キー コンテナーを作成してシークレットを追加する
-- MAGIC * Azure Databricks ワークスペースを作成してシークレット スコープを追加する
-- MAGIC 
-- MAGIC のパートを参照してください。(最後の"Azure Databricks から自分の BLOB コンテナーにアクセスする"はDBFSにマウントするシナリオになっていますので、不要です。)
-- MAGIC 
-- MAGIC [チュートリアル:Azure Key Vault を使用して Azure Databricks から Azure Blob Storage にアクセスする](https://docs.microsoft.com/ja-jp/azure/databricks/scenarios/store-secrets-azure-key-vault)
-- MAGIC 
-- MAGIC **注意:** Secretの登録はWebポータルUI上から可能ですが、Secretのリスト・削除はDatabricks CLIもしくはAPI経由での操作が必要になります。
-- MAGIC 
-- MAGIC 上記のチュートリアルと同じ名前のSecretsスコープを設定した場合に、ADLSにアクセスするサンプルは以下の通りになります。
-- MAGIC 
-- MAGIC (単に`access_key`の部分をSecretsから値を取得するように書き換えているだけです)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC 
-- MAGIC # ストレージアカウント、コンテナ、アクセスキーの設定
-- MAGIC storage_account = 'your_ADLS_storage_account'
-- MAGIC storage_container = 'your_ADLS_storage_container'
-- MAGIC #access_key = 'Your-ADLS-Container-Access-Key'
-- MAGIC access_key = dbutils.secrets.get(scope = "databricks-tutorial-secret-scope", key = "DbksStorageKey")
-- MAGIC 
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

-- MAGIC %python
-- MAGIC 
-- MAGIC # 上記のpathを使って、直接ADLS上のファイルにアクセスできます。
-- MAGIC # 例えば、上記のstorage container上にある`/foo/bar/example.csv`のファイルをpandasで読み込むコードは以下の通りになります。
-- MAGIC import pandas as pd
-- MAGIC df = pd.read_csv(f'abfss://{storage_container}@{storage_account}.dfs.core.windows.net/foo/bar/example.csv')
