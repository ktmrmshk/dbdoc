# Databricks notebook source
### 2. 上記のDeltaテーブルからサマリのDeltaテーブルを作る

df=spark.read.format('delta').load('/tmp/daiwt2021/job/tweet.delta')

(
  df.groupBy('lang').count()
  .write.format('delta').mode('overwrite').save('/tmp/daiwt2021/job/tweet_summary.delta')
)

sql("CREATE TABLE IF NOT EXISTS tweet_summary USING delta LOCATION '/tmp/daiwt2021/job/tweet_summary.delta'")

# 確認
display(
  spark.read.format('delta').load('/tmp/daiwt2021/job/tweet_summary.delta')
)
