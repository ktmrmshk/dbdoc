# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>Databricksで実践するData &amp; AI</h1>  
# MAGIC <h2>映画配信のレコメンデーションエンジン</h2>
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Shunichiro Takeshita</th></tr>
# MAGIC   <tr><td>日付</td><td>2020/09/01</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>DBR7.1ML</td></tr>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>ステップ 1: データセットをダウンロードしてDBFSに格納</h1>  
# MAGIC <p>・　wgetコマンドにてZipファイルをダウンロード</p>
# MAGIC <p>・　**%sh** にてSpark Drive上で実行</p>

# COMMAND ----------

# MAGIC %sh wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -O /tmp/ml-1m.zip

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Unzip the files.

# COMMAND ----------

# MAGIC %sh rm  -Rf /tmp/ml-1m/

# COMMAND ----------

# MAGIC %sh unzip /tmp/ml-1m.zip -d /tmp

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Copy the file to S3 so that it can be accessed by the Spark workers as well.

# COMMAND ----------

# MAGIC %fs cp -r file:/tmp/ml-1m/ dbfs:/tmp/stakeshita@databricks.com/movielens1m

# COMMAND ----------

# MAGIC %sh cat /tmp/ml-1m/README

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 2: Register the users data as a DataFrame.

# COMMAND ----------

# MAGIC %fs head dbfs:/tmp/stakeshita@databricks.com/movielens1m/users.dat

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP DATABASE IF EXISTS MovieDemo;
# MAGIC CREATE DATABASE MovieDemo;
# MAGIC USE MovieDemo;

# COMMAND ----------

# MAGIC %sql 
# MAGIC DROP TABLE IF EXISTS users;
# MAGIC 
# MAGIC CREATE EXTERNAL TABLE users (
# MAGIC   user_id INT,
# MAGIC   gender STRING,
# MAGIC   age INT,
# MAGIC   occupation_id INT,
# MAGIC   zipcode STRING
# MAGIC )
# MAGIC ROW FORMAT
# MAGIC   SERDE 'org.apache.hadoop.hive.serde2.RegexSerDe'
# MAGIC WITH SERDEPROPERTIES (
# MAGIC   'input.regex' = '([^:]+)::([^:]+)::([^:]+)::([^:]+)::([^:]+)'
# MAGIC )
# MAGIC LOCATION 
# MAGIC   'dbfs:/tmp/stakeshita@databricks.com/movielens1m/users*.dat'

# COMMAND ----------

# MAGIC %sql select * from users

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3: Register the ratings data as a DataFrame.

# COMMAND ----------

# MAGIC %fs head dbfs:/tmp/stakeshita@databricks.com/movielens1m/ratings.dat

# COMMAND ----------

from pyspark.sql import Row
import datetime

def create_row_for_rating(line):
  atoms = line.split("::")
  return Row(user_id = int(atoms[0]),
             movie_id = int(atoms[1]),
             rating = int(atoms[2]),
             timestamp = datetime.datetime.fromtimestamp(int(atoms[3])))

ratings = sc.textFile("dbfs:/tmp/stakeshita@databricks.com/movielens1m/ratings.dat").map(create_row_for_rating)
ratingsDF = sqlContext.createDataFrame(ratings)
#ratingsDF.registerTempTable("ratings")
ratingsDF.write.mode('overwrite').saveAsTable('ratings')

# COMMAND ----------

# MAGIC %sql select * from ratings

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 4: Import the movies data as a DataFrame.

# COMMAND ----------

# MAGIC %fs head dbfs:/tmp/stakeshita@databricks.com/movielens1m/movies.dat

# COMMAND ----------

from pyspark.sql import Row

def create_row(line):
  atoms = line.split("::")
  movie = {"id": int(atoms[0])}
  year_begin = atoms[1].rfind("(")
  movie["name"] = atoms[1][0:year_begin].strip()
  movie["year"] = int(atoms[1][year_begin+1:-1])
  movie["categories"] = atoms[2].split("|")
  return Row(**movie)
  
movies = sc.textFile("dbfs:/tmp/stakeshita@databricks.com/movielens1m/movies.dat").map(create_row)
moviesDF = sqlContext.createDataFrame(movies)
#moviesDF.registerTempTable("movies")
moviesDF.write.mode('overwrite').saveAsTable('movies')

# COMMAND ----------

# MAGIC %sql select id, name, year, categories from movies
