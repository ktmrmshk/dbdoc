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
# MAGIC <div style='line-height:1.5rem; padding-top: 10px;'>
# MAGIC <h2>ALSモデルを利用したレコメンデーションエンジン開発 ＋ 本番デプロイ</h2>
# MAGIC <p>*ALS : Alternating Least Squares</p>
# MAGIC 
# MAGIC   <div style='float:right; width:40%;'>
# MAGIC   <img src='https://psajpstorage.blob.core.windows.net/commonfiles/ALS001.png' style='padding-right:30px;' width='100%'>
# MAGIC </div>
# MAGIC 
# MAGIC <h2>本編内容</h2>
# MAGIC <p>今回は映画レイティングデータ、ユーザの好みを利用して推奨映画を導出します。</p>
# MAGIC <p></p>
# MAGIC ❶ Databricksウィジェットから好きな映画を選択。<br>
# MAGIC ❷ モデル学習と評価。<br>
# MAGIC ❸ ダッシュボードの作成。<br>
# MAGIC ❹ MLFLOWと本番デプロイ。<br>
# MAGIC </div>
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2020/09/01</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.0</td></tr>
# MAGIC   <tr><td>クラスター</td><td>DBR7.2ML</td></tr>
# MAGIC </table>
# MAGIC <img style="margin-top:25px;" src="https://jixjiadatabricks.blob.core.windows.net/images/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>❶　Databricksウィジェットから好きな映画を選択 (For ユーザーインプット)</h1>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC <div>
# MAGIC <p>・　広く知れ渡った映画リストから10本のランダムな映画を選択します。<br>
# MAGIC    ・　ユーザーがそれらの映画の評価を入力できるように、Databricks ウィジェットを作成します。</p>
# MAGIC <p  style="position:relative;width: 1100px;height: 300px;overflow: hidden;">
# MAGIC <img style="margin-top:25px;" src="https://psajpstorage.blob.core.windows.net/commonfiles/Widget001.gif"  width="1000" border="3" >
# MAGIC </div>
# MAGIC <div style="padding:10px 0;">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/Movie_Back_to_the_Future_Part_II.jpg" width="200px">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/Movie_JurassicPark.jpg" width="200px">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/Movie_Starwars.jpg" width="200px">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/Movie_Titanic.jpg" width="200px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Movieテーブル：映画一覧
# MAGIC %sql
# MAGIC USE MovieDemo;
# MAGIC select * from movies

# COMMAND ----------

from pyspark.sql.functions import explode
sqlContext.sql("select * from movies").select("id", "name", "year", explode("categories").alias("category")).registerTempTable("exploded_movies")

# COMMAND ----------

# DBTITLE 1,カテゴリー別の映画件数
# MAGIC %sql 
# MAGIC select year, category, count(*) as the_count from exploded_movies where year > 1980 group by year, category order by year asc, the_count desc

# COMMAND ----------

# DBTITLE 1,Raitingsテーブル：映画ごとの評価レイティング
# MAGIC %sql 
# MAGIC select movie_id, movies.name, movies.year, count(*) as times_rated from ratings join movies on ratings.movie_id = movies.id group by movie_id, movies.name, movies.year order by times_rated desc

# COMMAND ----------

sqlContext.sql("""
    select 
      movie_id, movies.name, count(*) as times_rated 
    from 
      ratings
    join 
      movies on ratings.movie_id = movies.id
    group by 
      movie_id, movies.name, movies.year
    order by 
      times_rated desc
    limit
      200
    """
).registerTempTable("most_rated_movies")

# COMMAND ----------

if not "most_rated_movies" in vars():
  most_rated_movies = sqlContext.table("most_rated_movies").rdd.takeSample(True, 10)
  for i in range(0, len(most_rated_movies)):
    dbutils.widgets.dropdown("movie_%i" % i, "5", ["1", "2", "3", "4", "5"], most_rated_movies[i].name)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h2>ウィジェットの結果をパラメータとして取得して、機械学習モデル作成へ</h2>

# COMMAND ----------

from datetime import datetime
from pyspark.sql import Row
ratings = []
for i in range(0, len(most_rated_movies)):
  ratings.append(
    Row(user_id = 0,
        movie_id = most_rated_movies[i].movie_id,
        rating = float(dbutils.widgets.get("movie_%i" %i)),
        timestamp = datetime.now()))
myRatingsDF = sqlContext.createDataFrame(ratings)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>❷　モデル学習と評価</h1>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h2>❷-1 学習データと検証データ作成</h2>

# COMMAND ----------

from pyspark.sql import functions

ratings = sqlContext.table("ratings")
ratings = ratings.withColumn("rating", ratings.rating.cast("float"))
(training, test) = ratings.randomSplit([0.8, 0.2])
print(f'学習サイズ:テストサイズ = {training.count()}:{test.count()}')

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h2>❷-2 レイティング・テーブルを利用してALSモデルを作成</h2>

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="movie_id", ratingCol="rating")
model = als.fit(training.unionAll(myRatingsDF))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h2>❷-3 RMSEを算出してモデルを評価</h2>

# COMMAND ----------

predictions = model.transform(test).dropna()
predictions.registerTempTable("predictions")

# COMMAND ----------

# MAGIC %sql select user_id, movie_id, rating, prediction from predictions

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
displayHTML("<h4>RMSE :  %s</h4>" % str(rmse))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h2>❷-4 最後にご自身の推奨映画を表示します</h2>

# COMMAND ----------

#ウィジェットの結果を反映
mySampledMovies = model.transform(myRatingsDF)
mySampledMovies.registerTempTable("mySampledMovies")

# COMMAND ----------

from pyspark.sql import functions
df = sqlContext.table("movies")
myGeneratedPredictions = model.transform(df.select(df.id.alias("movie_id")).withColumn("user_id", functions.expr("int('0')")))
myGeneratedPredictions.dropna().registerTempTable("myPredictions")

# COMMAND ----------

# DBTITLE 1,ご自身へのお薦め映画
# MAGIC %sql 
# MAGIC SELECT 
# MAGIC   /*categories,*/ most_rated_movies.name, case when prediction > 5 then '☆☆☆☆☆' when prediction > 4 then '☆☆☆☆' when prediction > 4 then '☆☆☆' else '☆☆' end as score
# MAGIC from 
# MAGIC   myPredictions 
# MAGIC join most_rated_movies on myPredictions.movie_id = most_rated_movies.movie_id
# MAGIC join movies on myPredictions.movie_id = movies.id
# MAGIC order by
# MAGIC   prediction desc
# MAGIC LIMIT
# MAGIC   10

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>❸ ダッシュボードの作成</h1>

# COMMAND ----------



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h1>❹ MLFLOWと本番デプロイ</h1>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <div style='float:right; width:30%;'>
# MAGIC   <img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" style='padding-right:300px;' width='100%'>
# MAGIC </div>
# MAGIC 
# MAGIC <h3>MLFlowにて実験結果を保存する</h3>  
# MAGIC ❶ 機械学習に特化したロギング用のAPI。<br>
# MAGIC ❷ 学習を行う環境やライブラリーに依存すること無くトラッキング可能。<br>
# MAGIC ❸ データサイエンスのコードを実行すること、即ち `runs` した情報を全てトラッキング。<br>
# MAGIC ❹ この `runs` の結果が `experiments` として集約されて情報化。<br>
# MAGIC ❺ MLflow サーバではこれらの情報を管理
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h3>作成したモデルを MLflow モデル・レジストリー に登録</h3>  
# MAGIC MLflow Model Registry は、モデルの一元管理を行い、機械学習の全てのライフサイクルを管理するための各種API機能やUIを持っています。<br>
# MAGIC モデルのリネージ(モデル構築から本番環境での実行まで)、モデルのバージョン管理、ステージの移行(ステージ環境から本番環境への移行)などを行います。</div>
# MAGIC 
# MAGIC <div style='float:Center; width:70%;'>
# MAGIC   <img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/mlflow-repository.png" style='padding-right:300px;' width='100%'>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <h3>Amazon SageMaker　にモデルをデプロイ</h3>  
# MAGIC デプロイの際、MLflow は、Dockerコンテナを用いてモデルをロードし設置します。Amazon ECR(Elastic Container Registry)にも登録が可能です。<br>
# MAGIC コンテナがデプロイされると、SageMakerインスタンスを起動するとモデルの利用が可能です。
# MAGIC </div>
# MAGIC 
# MAGIC <div style='float:Center; width:70%;'>
# MAGIC   <img src="https://github.com/HimanshuAroraDb/Images/blob/master/sgaemaker-ecr.png?raw=true" style='padding-right:300px;' width='100%'>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC 
# MAGIC <div><h1>END</h1></div>

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from movies

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC category, sum(rating) as rating
# MAGIC from ratings
# MAGIC join users on ratings.user_id = users.user_id
# MAGIC join exploded_movies on ratings.movie_id = exploded_movies.id
# MAGIC where users.gender = 'M' and users.age between 35 and 45
# MAGIC group by 1
# MAGIC order by 2 desc
# MAGIC limit 10

# COMMAND ----------

# MAGIC %sql
# MAGIC select category, name, case when rating > 4.5 then '☆☆☆☆☆' when rating > 4.2 then '☆☆☆☆' when rating > 4.0 then '☆☆☆' else '☆☆' end as rating
# MAGIC from (
# MAGIC select
# MAGIC category, name, avg(rating) as rating, sum(rating)
# MAGIC from ratings
# MAGIC join users on ratings.user_id = users.user_id
# MAGIC join exploded_movies on ratings.movie_id = exploded_movies.id
# MAGIC where users.gender = 'M' and users.age between 35 and 45
# MAGIC group by 1,2
# MAGIC having category = 'Drama'
# MAGIC order by 4 desc
# MAGIC Limit 10
# MAGIC ) a
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select category, name, case when rating > 4.5 then '☆☆☆☆☆' when rating > 4.2 then '☆☆☆☆' when rating > 4.0 then '☆☆☆' else '☆☆' end as rating
# MAGIC from (
# MAGIC select
# MAGIC category, name, avg(rating) as rating, sum(rating)
# MAGIC from ratings
# MAGIC join users on ratings.user_id = users.user_id
# MAGIC join exploded_movies on ratings.movie_id = exploded_movies.id
# MAGIC where users.gender = 'M' and users.age between 35 and 45
# MAGIC group by 1,2
# MAGIC having category = 'Comedy'
# MAGIC order by 4 desc
# MAGIC Limit 10
# MAGIC ) a
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <head>
# MAGIC   <link href="https://fonts.googleapis.com/css2?family=Kosugi+Maru&display=swap" rel="stylesheet">
# MAGIC   <style>
# MAGIC     h1,h2,h3,p,span,td, div {font-family: "Kosugi Maru", sans-serif !important;}
# MAGIC   </style>
# MAGIC </head>
# MAGIC <div style="padding:10px 0;">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/movie_patriot.jpeg" width="120px">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/movie_private_ryan.jpg" width="120px">
# MAGIC <img src="https://psajpstorage.blob.core.windows.net/commonfiles/movie_airforce.jpg" width="120px">
# MAGIC </div>
