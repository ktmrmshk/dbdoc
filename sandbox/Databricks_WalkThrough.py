# Databricks notebook source
# MAGIC %md ## Managed Service
# MAGIC ----
# MAGIC 
# MAGIC * ユーザー管理
# MAGIC * Cluster管理
# MAGIC * Notebook管理
# MAGIC * ライブラリ管理
# MAGIC * データ・テーブル管理
# MAGIC * 学習モデル管理

# COMMAND ----------

# MAGIC %md ## ETL with Spark

# COMMAND ----------

# 既存のデータを削除
dbutils.fs.rm('dbfs:/home/masahiko.kitamura@databricks.com/lending_club/', recurse=True)


# COMMAND ----------

# DBTITLE 1,ファイルの読み込み
# パス指定
#source_path = 'dbfs:/databricks-datasets/samples/lending_club/parquet/'
#parquet_path = 'dbfs:/home/parquet/lending-club-loan/'


# データを読み込む
df_raw = spark.read.parquet('dbfs:/databricks-datasets/samples/lending_club/parquet/')

# 読み込まれたデータを参照
display(df_raw)

# レコード数
print("レコード数=>", df_raw.count())


# randomSplit()を使って、5%のサンプルを読み取る
#(data, data_rest) = df.randomSplit([0.05, 0.95], seed=123)

# 読み込まれたデータを参照
#display(data)

# COMMAND ----------

# DBTITLE 1,ETL・データ加工・クレンジング(Sparkによるスケーラブル処理)
from pyspark.sql.functions import col, expr

# ETL処理(カラム抽出、データ変換、カラム名変更)
df_cleaned = (
  df_raw
  .select('loan_amnt', 
            'term',
            'int_rate',
            'grade',
            'addr_state',
            'emp_title',
            'home_ownership',
            'annual_inc',
            'loan_status')
  .withColumn('int_rate', expr('cast(replace(int_rate,"%","") as float)'))
  .withColumnRenamed('addr_state', 'state')
)

display(df_cleaned)

# COMMAND ----------

# DBTITLE 1,SQLからも参照できます (R, Scalaからも同様に参照できます)
# このdataframeをSQLから参照するために、temp viewを作成
df_cleaned.createOrReplaceTempView('view_cleaned')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT state, grade, avg(loan_amnt), count(1) FROM view_cleaned
# MAGIC WHERE loan_amnt > 10000
# MAGIC GROUP BY grade, state
# MAGIC ORDER BY count(1) desc

# COMMAND ----------

# DBTITLE 1,ファイルに書き出す(CSV)
df_cleaned.write.format('csv').mode('overwrite').save('dbfs:/home/masahiko.kitamura@databricks.com/lending_club.csv')

# COMMAND ----------

# MAGIC %md ### 従来のデータレイクの制限と限界
# MAGIC 
# MAGIC -----
# MAGIC * ファイルのパーティションをユーザーが管理しないといけない
# MAGIC * 細かいファイルがどんどん増えていく
# MAGIC * ファイル数が増えるにつれて読み込みに時間がかかる
# MAGIC * レコードは追記のみ(UPDATE, DELETE, MERGEができない)
# MAGIC * スキーマの整合性はユーザー側でチェックしないといけない
# MAGIC * 検索条件がパーティションキーでない場合、全てのファイルを開く必要がある
# MAGIC * Indexingなどの最適化機能がない
# MAGIC 
# MAGIC 
# MAGIC など。

# COMMAND ----------

# MAGIC %md ## Delta Lake
# MAGIC 
# MAGIC 
# MAGIC <hr>
# MAGIC <h3>データレイクに<span style="color='#38a'">信頼性</span>と<span style="color='#38a'">パフォーマンス</span>をもたらす</h3>
# MAGIC <p>本編はローン審査データを使用してDelta LakeでETLを行いながら、その主要機能に関して説明していきます。</p>
# MAGIC <div style="float:left; padding-right:60px; margin-top:20px; margin-bottom:200px;">
# MAGIC   <img src="https://jixjiadatabricks.blob.core.windows.net/images/delta-lake-square-black.jpg" width="220">
# MAGIC </div>
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

# DBTITLE 1,Delta Lake化(単にdelta形式として保存するだけ)
df_cleaned.write.format('delta').save('dbfs:/home/masahiko.kitamura@databricks.com/lending_club.delta')

# COMMAND ----------

# DBTITLE 1,Delta Lake化したファイルを読み込む
df_delta = spark.read.format('delta').load('dbfs:/home/masahiko.kitamura@databricks.com/lending_club.delta')
display(df_delta)

# SQLからも参照するためにテーブル作成(テーブル名とファイルパスの紐付け)
sql('''CREATE TABLE lbs USING delta LOCATION''')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Collaboration

# COMMAND ----------

# MAGIC %md ## Data Science

# COMMAND ----------

# MAGIC %md ## Realtime & Batch Data 

# COMMAND ----------


