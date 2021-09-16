# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # PythonからDeltaテーブルのデータ参照 (Databricks SQL Connector編)
# MAGIC 
# MAGIC ## 1. Installation
# MAGIC 
# MAGIC PyPIからインストール可能です。
# MAGIC 
# MAGIC ```
# MAGIC $ pip install databricks-sql-connector
# MAGIC ```
# MAGIC 
# MAGIC ## 2. 接続先クラスタ、トークンの確認
# MAGIC 
# MAGIC Databricks環境から以下の情報を取得します。
# MAGIC 
# MAGIC * Cluster
# MAGIC   - URL: 例: `xxxxx.cloud.databricks.com`
# MAGIC   - HTTP Path: 例: `sql/protocolv1/o/0/0804-2xxxxxxx9-stxxxxxx0`
# MAGIC 
# MAGIC * Personal Access Token
# MAGIC   - Databricks画面の右上にある`User Settings > Access Tokens`から発行できます。
# MAGIC   - 例: `dapi2ccxxxxxxxxxxxxxxxxxxxxxxx052e`
# MAGIC 
# MAGIC ## 3. サンプルCodeの実行
# MAGIC 
# MAGIC 
# MAGIC 以下が、Deltaテーブル(db名.table名) `usjdb.disp_joined` を参照するコードになります。
# MAGIC 単純のため、Tokenなどは平文で扱っていますので、実際にはセキュリティを考慮したコードに変更してください。
# MAGIC 
# MAGIC `db_sql_conn_sample.py`
# MAGIC ```python
# MAGIC from databricks import sql
# MAGIC 
# MAGIC conn = sql.connect(
# MAGIC     server_hostname = 'demo.cloud.databricks.com',
# MAGIC     http_path='sql/protocolv1/o/0/0804-220509-stead130',
# MAGIC     access_token='dapi2cc6fe7a0f73215d7733f8aba98005e3'
# MAGIC )
# MAGIC 
# MAGIC cur = conn.cursor()
# MAGIC cur.execute('SELECT * FROM usjdb.disp_joined limit 5')
# MAGIC 
# MAGIC ret = cur.fetchall()
# MAGIC 
# MAGIC for r in ret:
# MAGIC     print(r)
# MAGIC 
# MAGIC cur.close()
# MAGIC ```
# MAGIC 
# MAGIC 上記のコードを実行してみます。
# MAGIC 
# MAGIC ```
# MAGIC $ python db_sql_conn_sample.py
# MAGIC 
# MAGIC (datetime.datetime(2021, 8, 1, 2, 25), 'LEFT', 'TITLE_7', None, "ELMO'S GO GO SKATEBOARD エルモのゴーゴー・スケートボード", None, None, None)
# MAGIC (datetime.datetime(2021, 8, 1, 18, 25), 'OTHER', 'TITLE_2', '10 min', "ELMO'S LITTLE DRIVE エルモのリトル・ドライブ", '333333', '000000', None)
# MAGIC (datetime.datetime(2021, 8, 1, 0, 15), 'LEFT', 'TITLE_3', None, 'SING ON TOUR シング・オン・ツアー', '3333333333333333333333', '0000000000000000000000', None)
# MAGIC (datetime.datetime(2021, 8, 1, 12, 15), 'OTHER', 'TITLE_2', '5 min', "ELMO'S LITTLE DRIVE エルモのリトル・ドライブ", '33333', '00000', None)
# MAGIC (datetime.datetime(2021, 8, 1, 20, 15), 'OTHER', 'TITLE_25', None, 'OTHER 25', None, None, None)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # (参照)AWS Lambda上での実行
# MAGIC 
# MAGIC AWS Lambda上で実行するには、databricks-sql-connectorをLayerとして登録する必要があります。
# MAGIC その方法を見ていきます。
# MAGIC 
# MAGIC ## 1. databricks-sql-connectorをzipで固める
# MAGIC 
# MAGIC ```bash
# MAGIC $ mkdir python
# MAGIC $ cd python
# MAGIC $ pip install -t ./ databricks-sql-connector
# MAGIC $ zip -r Layer.zip python/
# MAGIC 
# MAGIC $ ls
# MAGIC Layer.zip  <==　これをAWS LambdaのLayerとしてアップロードする
# MAGIC ```
# MAGIC 
# MAGIC ## 2. AWS Lambda上にLayerとしてアップロードする
# MAGIC 
# MAGIC AWS CosoleからLambdaのLayerとして上記のzipファイルをアップロードし、Function実行のLayersに追加する。
# MAGIC 
# MAGIC ## 3. Functionコードを実行する
# MAGIC 
# MAGIC Layerの追加によって、
# MAGIC 上記のサンプルコードと同様に`from databricks import sql`でライブラリを読み込みが可能になる。

# COMMAND ----------


