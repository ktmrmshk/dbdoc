# Databricks notebook source
### 1. ストレージから更新ファイルだけを認識して、Deltaテーブルに追記する
df_autoloader = (
  spark.readStream.format('cloudFiles')
  .option('cloudFiles.format', 'json')
  .option('cloudFiles.maxBytesPerTrigger', '50KB')
  .schema(tweet_schema)
  .load('s3a://databricks-ktmr-s3/stocknet-dataset/tweet/raw/AAPL/*')
)

(
  df_autoloader.writeStream.format('delta')
  .option('checkpointLocation', '/tmp/daiwt2021/job/tweet.checkpoint')
  .option('maxFilesPerTrigger', 25)
  .outputMode('append')
  .trigger(once=True) # 一度だけ処理
  .start('/tmp/daiwt2021/job/tweet.delta')
  .awaitTermination() # async => sync
)
