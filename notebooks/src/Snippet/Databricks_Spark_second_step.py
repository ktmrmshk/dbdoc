# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC * Lakehouse(レイクハウス)とDatabricksの概要
# MAGIC * Sparkの基礎(dataframe)
# MAGIC * SQLの基礎
# MAGIC * Rの基礎
# MAGIC * ETLとDelta Lakeの基礎
# MAGIC * 練習問題

# COMMAND ----------

# MAGIC %md #Lakehouse(レイクハウス)とDatabricksの概要

# COMMAND ----------

slide_id = '18PJH5beIc7P4WD9TTiQIYXgDvHO8ptGU'
slide_number = '19'
displayHTML(f'''
<iframe
  src="https://docs.google.com/presentation/d/{slide_id}/embed?slide={slide_number}&rm=minimal"
  frameborder="0"
  width="100%"
  height="700"
></iframe>''')

# COMMAND ----------

# MAGIC %md # Sparkの基礎
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * DataFrameでデータを扱う
# MAGIC * Python, SQL, Scala, R、(Notebookを使用しなければ)Javaでコードを実装できる
# MAGIC * 並列分散環境(クラスタ)で処理が実行される
# MAGIC * Sparkがオープンソースのため、様々はデータソース・システムと連携が可能である
# MAGIC * pandasと同じ文法でspark処理が書けるkoalasというライブラリがある
# MAGIC * Batch処理に加えて、Streaming処理でもDataFrameで処理する
# MAGIC 
# MAGIC * databricks環境には、`/databricks-datasets/`配下にサンプルデータセットが用意されている

# COMMAND ----------

# MAGIC %python 
# MAGIC # このnotebookのデフォルト言語はPythonに指定されているので、セル上部の"%python"の記述は不要です。
# MAGIC 
# MAGIC # parquet
# MAGIC df = spark.read.format('parquet').load('dbfs:/databricks-datasets/samples/lending_club/parquet/')
# MAGIC display()

# COMMAND ----------



# COMMAND ----------

# MAGIC %
