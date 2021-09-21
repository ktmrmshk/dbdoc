# Databricks notebook source
# MAGIC %md # Init Scriptの使用例
# MAGIC 
# MAGIC Init Scriptを利用すると、Cluster起動時に特定のコマンドを自動で実行することができます。
# MAGIC 例えば、`apt`でパッケージを導入するなどの処理が自動化できます。
# MAGIC 
# MAGIC ここでは、Init Scriptで`apt install fortune`コマンドをクラスタ起動時に自動で実行する例を見ていきます。

# COMMAND ----------

# MAGIC %md ## Scriptの作成
# MAGIC 
# MAGIC (クラスタ範囲の)Init Scriptを実行するには実行する処理(シェルスクリプト)をファイルにして、DBFS(もしくはクラスタから参照できるストレージ, S3やAzure Blog Storageなど)に配置する必要があります。ここでは、DBFS上にスクリプトを配置します。
# MAGIC 
# MAGIC DBFSへのファイルのアップロードは`Databricks CLI`を使う必要があり、やや準びが必要になるので、シンプルにdatabricksのnotebook上でスクリプトファイルを作成し、それをDBFS上にコピーする方法をとります。

# COMMAND ----------

# DBTITLE 1,notebook上でスクリプトを作成(Driver nodeローカルに保存)
cmd='''
#!/bin/bash
apt update
apt install -y fortune
'''

with open('/tmp/install_fortune.sh', 'w') as f:
  f.write(cmd)


# COMMAND ----------

# MAGIC %sh
# MAGIC cat /tmp/install_fortune.sh

# COMMAND ----------

# DBTITLE 1,スクリプトをDriver nodeからDBFSにコピーする
# MAGIC %sh 
# MAGIC mkdir -p /dbfs/FileStore/init_scripts
# MAGIC cp /tmp/install_fortune.sh /dbfs/FileStore/init_scripts

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/init_scripts/

# COMMAND ----------

# MAGIC %md ## Clusterの設定
# MAGIC 
# MAGIC 1. Databricksの左メニュー一覧から`Compute`を選択し、Cluster設定画面に移動する。
# MAGIC 2. Cluster構成の`Advanced Option`の中の`Init Scripts`から上記でファイルを配置したDBFS上のパスを登録する。
# MAGIC 3. クラスタを起動する。
# MAGIC 
# MAGIC ![https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/cluster_init_scripts.png](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/image/cluster_init_scripts.png)

# COMMAND ----------

# DBTITLE 1,Init Srciptの動作確認 (コマンドがインストールされているか確認)
# MAGIC %sh
# MAGIC /usr/games/fortune
