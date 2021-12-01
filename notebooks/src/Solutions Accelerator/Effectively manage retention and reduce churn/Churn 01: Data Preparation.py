# Databricks notebook source
# MAGIC %md このノートブックの目的は、解約予測に必要なデータにアクセスして準備することです。 このノートブックは、**Databricks ML 7.1+**と**CPUベース**のノードを利用したクラスタ上で実行する必要があります。

# COMMAND ----------

# MAGIC %fs mount s3a://databricks-ktmr-s3/kkbox /mnt/kkbox

# COMMAND ----------

# MAGIC %fs ls /mnt/kkbox/

# COMMAND ----------

# MAGIC %md ###ステップ1: データの読み込み
# MAGIC 
# MAGIC 2018年、台湾に拠点を置く人気の音楽ストリーミングサービスである[KKBox](https://www.kkbox.com/)は、データ＆AIコミュニティが将来の期間にどの顧客が解約するかを予測することに挑戦することを目的として、2年強の（匿名化された）顧客のトランザクションとアクティビティのデータからなる[データセット](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)を公開しました。 
# MAGIC 
# MAGIC **注**データの提供条件により、この作品を再現するには、Kaggleからこのデータセットを構成するファイルをダウンロードし、以下のようなフォルダ構造を自分の環境で作成する必要があります。
# MAGIC 
# MAGIC ダウンロード可能な主要データファイルは、あらかじめ設定された[マウントポイント](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)の下に以下のように整理されています。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC これらのファイルをデータフレームに読み込むと、以下のようなデータモデルになります。
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_schema.png' width=300>
# MAGIC 
# MAGIC 各サービス加入者は、members テーブルの *msno* フィールドの値によって一意に識別されます。トランザクションテーブルとユーザーログテーブルのデータは、それぞれサブスクリプション管理とストリーミング活動の記録となります。 すべてのメンバーがこのスキーマのデータ一式を持っているわけではありません。 また、トランザクションおよびストリーミング・ログは、ある日のサブスクライバーに対して複数のレコードが記録されるため、非常に冗長である。 アクティビティがない日には、これらのテーブルにサブスクライバーのエントリはありません。
# MAGIC 
# MAGIC データのプライバシーを保護するために、これらのテーブルの多くの値は順序エンコードされており、その解釈は制限されています。 さらに、タイムスタンプ情報は日単位で切り捨てられているため、特定の日付のレコードの順序付けは、このノートブックの後のステップで扱うビジネスロジックに依存します。
# MAGIC  
# MAGIC それでは、このデータをロードしてみましょう。

# COMMAND ----------

# DBTITLE 1,ライブラリとコンフィグ環境のインポート
import shutil
from datetime import date

from pyspark.sql.types import *
from pyspark.sql.functions import lit

# COMMAND ----------

# DBTITLE 1,Trueに設定すると、メンバー、トランザクション、ユーザーログのテーブルの再読み込みがスキップされます。
# これは、次のようなシナリオのために追加されました。
# チャーンラベルの予測ロジックの一部を変更したいが、
# ノートブック全体を再実行したくないというシナリオのために追加されました。
skip_reload = False

# COMMAND ----------

# DBTITLE 1,テーブルを含むデータベースの作成
# create database to house SQL tables
_ = spark.sql('CREATE DATABASE IF NOT EXISTS kkbox')

# COMMAND ----------

# DBTITLE 1,Membersテーブルの読み込み
if not skip_reload:
  
  # delete the old table if needed
  _ = spark.sql('DROP TABLE IF EXISTS kkbox.members')

  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/mnt/kkbox/silver/members', ignore_errors=True)

  # members dataset schema
  member_schema = StructType([
    StructField('msno', StringType()),
    StructField('city', IntegerType()),
    StructField('bd', IntegerType()),
    StructField('gender', StringType()),
    StructField('registered_via', IntegerType()),
    StructField('registration_init_time', DateType())
    ])

  # read data from csv
  members = (
    spark
      .read
      .csv(
        '/mnt/kkbox/members/members_v3.csv',
        schema=member_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  (
    members
      .write
      .format('delta')
      .mode('overwrite')
      .save('/mnt/kkbox/silver/members')
    )

    # create table object to make delta lake queriable
  _ = spark.sql('''
      CREATE TABLE kkbox.members 
      USING DELTA 
      LOCATION '/mnt/kkbox/silver/members'
      ''')

# COMMAND ----------

# DBTITLE 1,Transactionsテーブルの読み込み
if not skip_reload:
  
# delete the old database and tables if needed
  _ = spark.sql('DROP TABLE IF EXISTS kkbox.transactions')

  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/mnt/kkbox/silver/transactions', ignore_errors=True)

  # transaction dataset schema
  transaction_schema = StructType([
    StructField('msno', StringType()),
    StructField('payment_method_id', IntegerType()),
    StructField('payment_plan_days', IntegerType()),
    StructField('plan_list_price', IntegerType()),
    StructField('actual_amount_paid', IntegerType()),
    StructField('is_auto_renew', IntegerType()),
    StructField('transaction_date', DateType()),
    StructField('membership_expire_date', DateType()),
    StructField('is_cancel', IntegerType())  
    ])

  # read data from csv
  transactions = (
    spark
      .read
      .csv(
        '/mnt/kkbox/transactions',
        schema=transaction_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  ( transactions
      .write
      .format('delta')
      .partitionBy('transaction_date')
      .mode('overwrite')
      .save('/mnt/kkbox/silver/transactions')
    )

    # create table object to make delta lake queriable
  _ = spark.sql('''
      CREATE TABLE kkbox.transactions
      USING DELTA 
      LOCATION '/mnt/kkbox/silver/transactions'
      ''')

# COMMAND ----------

# DBTITLE 1,Load User Logs Table
 Tableif not skip_reload:
  # delete the old table if needed
  _ = spark.sql('DROP TABLE IF EXISTS kkbox.user_logs')

  # drop any old delta lake files that might have been created
  shutil.rmtree('/dbfs/mnt/kkbox/silver/user_logs', ignore_errors=True)

  # transaction dataset schema
  user_logs_schema = StructType([ 
    StructField('msno', StringType()),
    StructField('date', DateType()),
    StructField('num_25', IntegerType()),
    StructField('num_50', IntegerType()),
    StructField('num_75', IntegerType()),
    StructField('num_985', IntegerType()),
    StructField('num_100', IntegerType()),
    StructField('num_uniq', IntegerType()),
    StructField('total_secs', FloatType())  
    ])

  # read data from csv
  user_logs = (
    spark
      .read
      .csv(
        '/mnt/kkbox/user_logs',
        schema=user_logs_schema,
        header=True,
        dateFormat='yyyyMMdd'
        )
      )

  # persist in delta lake format
  ( user_logs
      .write
      .format('delta')
      .partitionBy('date')
      .mode('overwrite')
      .save('/mnt/kkbox/silver/user_logs')
    )

  # create table object to make delta lake queriable
  _ = spark.sql('''
    CREATE TABLE IF NOT EXISTS kkbox.user_logs
    USING DELTA 
    LOCATION '/mnt/kkbox/silver/user_logs'
    ''')

# COMMAND ----------

# MAGIC %md ###ステップ2: チャーンラベルの取得
# MAGIC 
# MAGIC モデルを構築するためには、対象となる2つの期間内にどの顧客が解約したかを特定する必要があります。 この期間は2017年2月と2017年3月です。 2017年2月に解約を予測するモデルを学習し、2017年3月に解約を予測するモデルの能力を評価するため、それぞれを学習データセットとテストデータセットとします。
# MAGIC 
# MAGIC Kaggleコンペティションで提供された指示によると、KKBoxのサブスクライバは、サブスクリプションの有効期限が切れてから30日後に更新に失敗するまで、解約とは認識されません。 ほとんどのサブスクリプションは、それ自体が30日の更新スケジュールになっています（ただし、かなり長いサイクルで更新するサブスクリプションもあります）。つまり、解約を特定するには、顧客データを順に見ていき、顧客が以前の有効期限で解約したことを示す更新ギャップを探す必要があります。
# MAGIC 
# MAGIC 本コンテストでは、事前にラベル付けされたトレーニングデータとテストデータ（それぞれ*train.csv*と*train_v2.csv*）を提供していますが、過去の参加者からは、これらのデータセットを再生成する必要があるとの指摘がありました。 これを行うためのScalaスクリプトがKKBoxから提供されています。 このスクリプトを今回の環境に合わせて変更すると、以下のようにトレーニングデータセットとテストデータセットを再生成することができます。

# COMMAND ----------

# DBTITLE 1,トレーニングラベルの削除（存在する場合）
_ = spark.sql('DROP TABLE IF EXISTS kkbox.train')

shutil.rmtree('/dbfs/mnt/kkbox/silver/train', ignore_errors=True)

# COMMAND ----------

# DBTITLE 1,トレーニングラベルの生成（ロジックはKKBoxが提供）
# MAGIC %scala
# MAGIC 
# MAGIC import java.time.{LocalDate}
# MAGIC import java.time.format.DateTimeFormatter
# MAGIC import java.time.temporal.ChronoUnit
# MAGIC 
# MAGIC import org.apache.spark.sql.{Row, SparkSession}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import scala.collection.mutable
# MAGIC 
# MAGIC def calculateLastday(wrappedArray: mutable.WrappedArray[Row]) :String ={
# MAGIC   val orderedList = wrappedArray.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same plan, always subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expiration date keeps extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same day same plan transaction: subscription preceeds cancellation
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC   orderedList.last.getAs[String]("membership_expire_date")
# MAGIC }
# MAGIC 
# MAGIC def calculateRenewalGap(log:mutable.WrappedArray[Row], lastExpiration: String): Int = {
# MAGIC   val orderedDates = log.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same data same plan transaction, assumption: subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel of same plan, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expire date keep extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same date cancel should follow subscription
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC 
# MAGIC   //Search for the first subscription after expiration
# MAGIC   //If active cancel is the first action, find the gap between the cancellation and renewal
# MAGIC   val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
# MAGIC   var lastExpireDate = LocalDate.parse(s"${lastExpiration.substring(0,4)}-${lastExpiration.substring(4,6)}-${lastExpiration.substring(6,8)}", formatter)
# MAGIC   var gap = 9999
# MAGIC   for(
# MAGIC     date <- orderedDates
# MAGIC     if gap == 9999
# MAGIC   ) {
# MAGIC     val transString = date.getAs[String]("transaction_date")
# MAGIC     val transDate = LocalDate.parse(s"${transString.substring(0,4)}-${transString.substring(4,6)}-${transString.substring(6,8)}", formatter)
# MAGIC     val expireString = date.getAs[String]("membership_expire_date")
# MAGIC     val expireDate = LocalDate.parse(s"${expireString.substring(0,4)}-${expireString.substring(4,6)}-${expireString.substring(6,8)}", formatter)
# MAGIC     val isCancel = date.getAs[String]("is_cancel")
# MAGIC 
# MAGIC     if(isCancel == "1") {
# MAGIC       if(expireDate.isBefore(lastExpireDate)) {
# MAGIC         lastExpireDate = expireDate
# MAGIC       }
# MAGIC     } else {
# MAGIC       gap = ChronoUnit.DAYS.between(lastExpireDate, transDate).toInt
# MAGIC     }
# MAGIC   }
# MAGIC   gap
# MAGIC }
# MAGIC 
# MAGIC val data = spark
# MAGIC   .read
# MAGIC   .option("header", value = true)
# MAGIC   .csv("/mnt/kkbox/transactions/")
# MAGIC 
# MAGIC val historyCutoff = "20170131"
# MAGIC 
# MAGIC val historyData = data.filter(col("transaction_date")>="20170101" and col("transaction_date")<=lit(historyCutoff))
# MAGIC val futureData = data.filter(col("transaction_date") > lit(historyCutoff))
# MAGIC 
# MAGIC val calculateLastdayUDF = udf(calculateLastday _)
# MAGIC val userExpire = historyData
# MAGIC   .groupBy("msno")
# MAGIC   .agg(
# MAGIC     calculateLastdayUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       )
# MAGIC     ).alias("last_expire")
# MAGIC   )
# MAGIC 
# MAGIC val predictionCandidates = userExpire
# MAGIC   .filter(
# MAGIC     col("last_expire") >= "20170201" and col("last_expire") <= "20170228"
# MAGIC   )
# MAGIC   .select("msno", "last_expire")
# MAGIC 
# MAGIC 
# MAGIC val joinedData = predictionCandidates
# MAGIC   .join(futureData,Seq("msno"), "left_outer")
# MAGIC 
# MAGIC val noActivity = joinedData
# MAGIC   .filter(col("payment_method_id").isNull)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC 
# MAGIC val calculateRenewalGapUDF = udf(calculateRenewalGap _)
# MAGIC val renewals = joinedData
# MAGIC   .filter(col("payment_method_id").isNotNull)
# MAGIC   .groupBy("msno", "last_expire")
# MAGIC   .agg(
# MAGIC     calculateRenewalGapUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       ),
# MAGIC       col("last_expire")
# MAGIC     ).alias("gap")
# MAGIC   )
# MAGIC 
# MAGIC val validRenewals = renewals.filter(col("gap") < 30)
# MAGIC   .withColumn("is_churn", lit(0))
# MAGIC val lateRenewals = renewals.filter(col("gap") >= 30)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC val resultSet = validRenewals
# MAGIC   .select("msno","is_churn")
# MAGIC   .union(
# MAGIC     lateRenewals
# MAGIC       .select("msno","is_churn")
# MAGIC       .union(
# MAGIC         noActivity.select("msno","is_churn")
# MAGIC       )
# MAGIC   )
# MAGIC 
# MAGIC resultSet.write.format("delta").mode("overwrite").save("/mnt/kkbox/silver/train/")

# COMMAND ----------

# DBTITLE 1,Trainingラベルにアクセス
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE kkbox.train
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/kkbox/silver/train/';
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.train;

# COMMAND ----------

# DBTITLE 1,テストラベルの削除（存在する場合）
_ = spark.sql('DROP TABLE IF EXISTS kkbox.test')

shutil.rmtree('/dbfs/mnt/kkbox/silver/test', ignore_errors=True)

# COMMAND ----------

# DBTITLE 1,テストラベルの生成（KKBox提供のロジック）
# MAGIC %scala
# MAGIC 
# MAGIC import java.time.{LocalDate}
# MAGIC import java.time.format.DateTimeFormatter
# MAGIC import java.time.temporal.ChronoUnit
# MAGIC 
# MAGIC import org.apache.spark.sql.{Row, SparkSession}
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import scala.collection.mutable
# MAGIC 
# MAGIC def calculateLastday(wrappedArray: mutable.WrappedArray[Row]) :String ={
# MAGIC   val orderedList = wrappedArray.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same plan, always subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expiration date keeps extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same day same plan transaction: subscription preceeds cancellation
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC   orderedList.last.getAs[String]("membership_expire_date")
# MAGIC }
# MAGIC 
# MAGIC def calculateRenewalGap(log:mutable.WrappedArray[Row], lastExpiration: String): Int = {
# MAGIC   val orderedDates = log.sortWith((x:Row, y:Row) => {
# MAGIC     if(x.getAs[String]("transaction_date") != y.getAs[String]("transaction_date")) {
# MAGIC       x.getAs[String]("transaction_date") < y.getAs[String]("transaction_date")
# MAGIC     } else {
# MAGIC       
# MAGIC       val x_sig = x.getAs[String]("plan_list_price") +
# MAGIC         x.getAs[String]("payment_plan_days") +
# MAGIC         x.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       val y_sig = y.getAs[String]("plan_list_price") +
# MAGIC         y.getAs[String]("payment_plan_days") +
# MAGIC         y.getAs[String]("payment_method_id")
# MAGIC 
# MAGIC       //same data same plan transaction, assumption: subscribe then unsubscribe
# MAGIC       if(x_sig != y_sig) {
# MAGIC         x_sig > y_sig
# MAGIC       } else {
# MAGIC         if(x.getAs[String]("is_cancel")== "1" && y.getAs[String]("is_cancel") == "1") {
# MAGIC           //multiple cancel of same plan, consecutive cancels should only put the expiration date earlier
# MAGIC           x.getAs[String]("membership_expire_date") > y.getAs[String]("membership_expire_date")
# MAGIC         } else if(x.getAs[String]("is_cancel")== "0" && y.getAs[String]("is_cancel") == "0") {
# MAGIC           //multiple renewal, expire date keep extending
# MAGIC           x.getAs[String]("membership_expire_date") < y.getAs[String]("membership_expire_date")
# MAGIC         } else {
# MAGIC           //same date cancel should follow subscription
# MAGIC           x.getAs[String]("is_cancel") < y.getAs[String]("is_cancel")
# MAGIC         }
# MAGIC       }
# MAGIC     }
# MAGIC   })
# MAGIC 
# MAGIC   //Search for the first subscription after expiration
# MAGIC   //If active cancel is the first action, find the gap between the cancellation and renewal
# MAGIC   val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")
# MAGIC   var lastExpireDate = LocalDate.parse(s"${lastExpiration.substring(0,4)}-${lastExpiration.substring(4,6)}-${lastExpiration.substring(6,8)}", formatter)
# MAGIC   var gap = 9999
# MAGIC   for(
# MAGIC     date <- orderedDates
# MAGIC     if gap == 9999
# MAGIC   ) {
# MAGIC     val transString = date.getAs[String]("transaction_date")
# MAGIC     val transDate = LocalDate.parse(s"${transString.substring(0,4)}-${transString.substring(4,6)}-${transString.substring(6,8)}", formatter)
# MAGIC     val expireString = date.getAs[String]("membership_expire_date")
# MAGIC     val expireDate = LocalDate.parse(s"${expireString.substring(0,4)}-${expireString.substring(4,6)}-${expireString.substring(6,8)}", formatter)
# MAGIC     val isCancel = date.getAs[String]("is_cancel")
# MAGIC 
# MAGIC     if(isCancel == "1") {
# MAGIC       if(expireDate.isBefore(lastExpireDate)) {
# MAGIC         lastExpireDate = expireDate
# MAGIC       }
# MAGIC     } else {
# MAGIC       gap = ChronoUnit.DAYS.between(lastExpireDate, transDate).toInt
# MAGIC     }
# MAGIC   }
# MAGIC   gap
# MAGIC }
# MAGIC 
# MAGIC val data = spark
# MAGIC   .read
# MAGIC   .option("header", value = true)
# MAGIC   .csv("/mnt/kkbox/transactions/")
# MAGIC 
# MAGIC val historyCutoff = "20170228"
# MAGIC 
# MAGIC val historyData = data.filter(col("transaction_date")>="20170201" and col("transaction_date")<=lit(historyCutoff))
# MAGIC val futureData = data.filter(col("transaction_date") > lit(historyCutoff))
# MAGIC 
# MAGIC val calculateLastdayUDF = udf(calculateLastday _)
# MAGIC val userExpire = historyData
# MAGIC   .groupBy("msno")
# MAGIC   .agg(
# MAGIC     calculateLastdayUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       )
# MAGIC     ).alias("last_expire")
# MAGIC   )
# MAGIC 
# MAGIC val predictionCandidates = userExpire
# MAGIC   .filter(
# MAGIC     col("last_expire") >= "20170301" and col("last_expire") <= "20170331"
# MAGIC   )
# MAGIC   .select("msno", "last_expire")
# MAGIC 
# MAGIC 
# MAGIC val joinedData = predictionCandidates
# MAGIC   .join(futureData,Seq("msno"), "left_outer")
# MAGIC 
# MAGIC val noActivity = joinedData
# MAGIC   .filter(col("payment_method_id").isNull)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC 
# MAGIC val calculateRenewalGapUDF = udf(calculateRenewalGap _)
# MAGIC val renewals = joinedData
# MAGIC   .filter(col("payment_method_id").isNotNull)
# MAGIC   .groupBy("msno", "last_expire")
# MAGIC   .agg(
# MAGIC     calculateRenewalGapUDF(
# MAGIC       collect_list(
# MAGIC         struct(
# MAGIC           col("payment_method_id"),
# MAGIC           col("payment_plan_days"),
# MAGIC           col("plan_list_price"),
# MAGIC           col("transaction_date"),
# MAGIC           col("membership_expire_date"),
# MAGIC           col("is_cancel")
# MAGIC         )
# MAGIC       ),
# MAGIC       col("last_expire")
# MAGIC     ).alias("gap")
# MAGIC   )
# MAGIC 
# MAGIC val validRenewals = renewals.filter(col("gap") < 30)
# MAGIC   .withColumn("is_churn", lit(0))
# MAGIC val lateRenewals = renewals.filter(col("gap") >= 30)
# MAGIC   .withColumn("is_churn", lit(1))
# MAGIC 
# MAGIC val resultSet = validRenewals
# MAGIC   .select("msno","is_churn")
# MAGIC   .union(
# MAGIC     lateRenewals
# MAGIC       .select("msno","is_churn")
# MAGIC       .union(
# MAGIC         noActivity.select("msno","is_churn")
# MAGIC       )
# MAGIC   )
# MAGIC 
# MAGIC resultSet.write.format("delta").mode("overwrite").save("/mnt/kkbox/silver/test/")

# COMMAND ----------

# DBTITLE 1,Access Testing Labels
# MAGIC %sql
# MAGIC 
# MAGIC CREATE TABLE kkbox.test
# MAGIC USING DELTA
# MAGIC LOCATION '/mnt/kkbox/silver/test/';
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.test;

# COMMAND ----------

# MAGIC %md ##Step 3: トランザクションログのクリーンアップと強化
# MAGIC 
# MAGIC KKBoxが提供する解約スクリプト（前のステップで使用）では、解約の状態を判断するためにトランザクションイベント間の時間が使用されます。ある日に複数のトランザクションが記録されている場合、どのトランザクションがその日のアカウントの最終的な状態を表しているかを判断するために、複雑なロジックが使用されます。 このロジックでは、特定の日に特定のサブスクライバーに対して複数のトランザクションがある場合、以下のようにします。
# MAGIC 
# MAGIC 1. plan_list_price、payment_plan_days、payment_method_id の値を連結し、これらの値のうち「大きい」ものを他のものより前にあるものとみなす<br>
# MAGIC 2. 最後のステップで定義した連結された値が、この日付のレコード間で同じであれば、キャンセル、つまりis_cancel=1のレコードは、他のトランザクションの後に続くべきである<br>
# MAGIC 3. このシーケンスで複数のキャンセルがあった場合、有効期限が最も早いレコードがこの取引日の最後のレコードとなる<br>
# MAGIC 4. この一連の流れの中で、キャンセルはないが、キャンセル以外のものが複数ある場合、有効期限が最も新しいキャンセル以外のレコードが、その取引日の最後のレコードとなる<br>
# MAGIC 
# MAGIC このロジックをSQLに書き換えることで、各日付の最後のレコードを含むトランザクションログのクレンジングされたバージョンを生成することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.transactions_clean;
# MAGIC 
# MAGIC CREATE TABLE kkbox.transactions_clean
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH 
# MAGIC     transaction_sequenced (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         transaction_date,
# MAGIC         plan_list_price,
# MAGIC         payment_plan_days,
# MAGIC         payment_method_id,
# MAGIC         is_cancel,
# MAGIC         membership_expire_date,
# MAGIC         RANK() OVER (PARTITION BY msno, transaction_date ORDER BY plan_sort DESC, is_cancel) as sort_id  -- calc rank on price, days & method sort followed by cancel sort
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           msno,
# MAGIC           transaction_date,
# MAGIC           plan_list_price,
# MAGIC           payment_plan_days,
# MAGIC           payment_method_id,
# MAGIC           CONCAT(CAST(plan_list_price as string), CAST(payment_plan_days as string), CAST(payment_method_id as string)) as plan_sort,
# MAGIC           is_cancel,
# MAGIC           membership_expire_date
# MAGIC         FROM kkbox.transactions
# MAGIC         )
# MAGIC       )
# MAGIC   SELECT
# MAGIC     p.msno,
# MAGIC     p.transaction_date,
# MAGIC     p.plan_list_price,
# MAGIC     p.actual_amount_paid,
# MAGIC     p.plan_list_price - p.actual_amount_paid as discount,
# MAGIC     p.payment_plan_days,
# MAGIC     p.payment_method_id,
# MAGIC     p.is_cancel,
# MAGIC     p.is_auto_renew,
# MAGIC     p.membership_expire_date
# MAGIC   FROM kkbox.transactions p
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       x.msno,
# MAGIC       x.transaction_date,
# MAGIC       x.plan_list_price,
# MAGIC       x.payment_plan_days,
# MAGIC       x.payment_method_id,
# MAGIC       x.is_cancel,
# MAGIC       CASE   -- if is_cancel is 0 in last record then go with max membership date identified, otherwise go with lowest membership date
# MAGIC         WHEN x.is_cancel=0 THEN MAX(x.membership_expire_date)
# MAGIC         ELSE MIN(x.membership_expire_date)
# MAGIC         END as membership_expire_date
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         a.transaction_date,
# MAGIC         a.plan_list_price,
# MAGIC         a.payment_plan_days,
# MAGIC         a.payment_method_id,
# MAGIC         a.is_cancel,
# MAGIC         a.membership_expire_date
# MAGIC       FROM transaction_sequenced a
# MAGIC       INNER JOIN (
# MAGIC         SELECT msno, transaction_date, MAX(sort_id) as max_sort_id -- find last entries on a given date
# MAGIC         FROM transaction_sequenced 
# MAGIC         GROUP BY msno, transaction_date
# MAGIC         ) b
# MAGIC         ON a.msno=b.msno AND a.transaction_date=b.transaction_date AND a.sort_id=b.max_sort_id
# MAGIC         ) x
# MAGIC     GROUP BY 
# MAGIC       x.msno, 
# MAGIC       x.transaction_date, 
# MAGIC       x.plan_list_price,
# MAGIC       x.payment_plan_days,
# MAGIC       x.payment_method_id,
# MAGIC       x.is_cancel
# MAGIC    ) q
# MAGIC    ON 
# MAGIC      p.msno=q.msno AND 
# MAGIC      p.transaction_date=q.transaction_date AND 
# MAGIC      p.plan_list_price=q.plan_list_price AND 
# MAGIC      p.payment_plan_days=q.payment_plan_days AND 
# MAGIC      p.payment_method_id=q.payment_method_id AND 
# MAGIC      p.is_cancel=q.is_cancel AND 
# MAGIC      p.membership_expire_date=q.membership_expire_date;
# MAGIC      
# MAGIC SELECT * 
# MAGIC FROM kkbox.transactions_clean
# MAGIC ORDER BY msno, transaction_date;

# COMMAND ----------

# MAGIC %md この *クリーニングされた* トランザクションデータを使用すると、Scala コードにある 30 日間のギャップロジックを使用して、購読の開始と終了をより簡単に特定することができます。 このデータセットで表される2年以上の期間において、多くの購読者が解約し、解約した購読者の多くが再購読することに注意する必要があります。 この点を考慮して、異なる購読を識別するために購読IDを生成し、それぞれの購読者の開始日と終了日が重ならないようにします。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.subscription_windows;
# MAGIC 
# MAGIC CREATE TABLE kkbox.subscription_windows 
# MAGIC USING delta
# MAGIC AS
# MAGIC   WITH end_dates (
# MAGIC       SELECT p.*
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           m.msno,
# MAGIC           m.transaction_date,
# MAGIC           m.membership_expire_date,
# MAGIC           m.next_transaction_date,
# MAGIC           CASE
# MAGIC             WHEN m.next_transaction_date IS NULL THEN 1
# MAGIC             WHEN DATEDIFF(m.next_transaction_date, m.membership_expire_date) > 30 THEN 1
# MAGIC             ELSE 0
# MAGIC             END as end_flag,
# MAGIC           CASE
# MAGIC             WHEN m.next_transaction_date IS NULL THEN m.membership_expire_date
# MAGIC             WHEN DATEDIFF(m.next_transaction_date, m.membership_expire_date) > 30 THEN m.membership_expire_date
# MAGIC             ELSE DATE_ADD(m.next_transaction_date, -1)  -- then just move the needle to just prior to the next transaction
# MAGIC             END as end_date
# MAGIC         FROM (
# MAGIC           SELECT
# MAGIC             x.msno,
# MAGIC             x.transaction_date,
# MAGIC             CASE  -- correct backdated expirations for subscription end calculations
# MAGIC               WHEN x.membership_expire_date < x.transaction_date THEN x.transaction_date
# MAGIC               ELSE x.membership_expire_date
# MAGIC               END as membership_expire_date,
# MAGIC             LEAD(x.transaction_date, 1) OVER (PARTITION BY x.msno ORDER BY x.transaction_date) as next_transaction_date
# MAGIC           FROM kkbox.transactions_clean x
# MAGIC           ) m
# MAGIC         ) p
# MAGIC       WHERE p.end_flag=1
# MAGIC     )
# MAGIC   SELECT
# MAGIC     ROW_NUMBER() OVER (ORDER BY subscription_start, msno) as subscription_id,
# MAGIC     msno,
# MAGIC     subscription_start,
# MAGIC     subscription_end
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       x.msno,
# MAGIC       MIN(x.transaction_date) as subscription_start,
# MAGIC       y.window_end as subscription_end
# MAGIC     FROM kkbox.transactions_clean x
# MAGIC     INNER JOIN (
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         COALESCE( MAX(b.end_date), '2015-01-01') as window_start,
# MAGIC         a.end_date as window_end
# MAGIC       FROM end_dates a
# MAGIC       LEFT OUTER JOIN end_dates b
# MAGIC         ON a.msno=b.msno AND a.end_date > b.end_date
# MAGIC       GROUP BY a.msno, a.end_date
# MAGIC       ) y
# MAGIC       ON x.msno=y.msno AND x.transaction_date BETWEEN y.window_start AND y.window_end
# MAGIC     GROUP BY x.msno, y.window_end
# MAGIC     )
# MAGIC   ORDER BY subscription_id;
# MAGIC   
# MAGIC SELECT *
# MAGIC FROM kkbox.subscription_windows
# MAGIC ORDER BY subscription_id;

# COMMAND ----------

# MAGIC %md 2017年2月と3月に解約のリスクがある顧客を特定するために使用されたスクリプトとサブスクリプションウィンドウが一致していることを確認するために、簡単なテストを行ってみましょう。 このスクリプトでは、履歴期間（対象月の開始までの期間）に記録された最後のトランザクションの有効期限が、対象期間の開始までの30日間とその期間の終了までの間にあるものをリスクのある契約としています。 例えば、2017年2月のリスクのあるお客様を特定するとしたら、2017年2月1日から2017年2月28日までの30日間に有効期限が設定されているアクティブなサブスクリプションを持つお客様を探す必要があります。 このシフトされたウィンドウは、30日間の猶予期間が関心のある期間内に失効するための時間を確保します。
# MAGIC 
# MAGIC **注意** より良いロジックは、期間開始の30日前と期間終了の30日前の間に有効期限がある契約に評価を限定することです。 (このような論理では、対象期間内に有効期限が切れるが、期間終了後まで30日間の猶予期間を終了しない契約は除外される)。このロジックを使用すると、提供されたスクリプトではリスクがあると認識されるが、当社では認識しない契約が多数見つかりました。 この演習では、競合他社のロジックに合わせることにします。
# MAGIC 
# MAGIC このロジックを念頭に置いて、リスクありと表示されたお客様がすべてこのロジックに従っているかどうかを確認してみましょう。
# MAGIC 
# MAGIC **注意** 私たちのロジックが有効であれば、次の2つのセルは結果を返さないはずです。

# COMMAND ----------

# DBTITLE 1,トレーニングデータセットに含まれる、リスクがないと思われるサブスクリプションの特定
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno
# MAGIC FROM kkbox.train x
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT DISTINCT -- subscriptions that had risk in Feb 2017
# MAGIC     a.msno
# MAGIC   FROM kkbox.subscription_windows a
# MAGIC   INNER JOIN kkbox.transactions_clean b
# MAGIC     ON a.msno=b.msno AND b.transaction_date BETWEEN a.subscription_start AND a.subscription_end
# MAGIC   WHERE 
# MAGIC         a.subscription_start < '2017-02-01' AND
# MAGIC         (b.membership_expire_date BETWEEN DATE_ADD('2017-02-01',-30) AND '2017-02-28')
# MAGIC   ) y
# MAGIC   ON x.msno=y.msno
# MAGIC WHERE y.msno IS NULL

# COMMAND ----------

# DBTITLE 1,リスクがないと思われるテストデータセット内のサブスクリプションの特定
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno
# MAGIC FROM kkbox.test x
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT DISTINCT -- subscriptions that had risk in Feb 2017
# MAGIC     a.msno
# MAGIC   FROM kkbox.subscription_windows a
# MAGIC   INNER JOIN kkbox.transactions_clean b
# MAGIC     ON a.msno=b.msno AND b.transaction_date BETWEEN a.subscription_start AND a.subscription_end
# MAGIC   WHERE 
# MAGIC         a.subscription_start < '2017-03-01' AND
# MAGIC         (b.membership_expire_date BETWEEN DATE_ADD('2017-03-01',-30) AND '2017-03-31')
# MAGIC   ) y
# MAGIC   ON x.msno=y.msno
# MAGIC WHERE y.msno IS NULL

# COMMAND ----------

# MAGIC %md 我々は、提供されたスクリプトと同じリスクのあるサブスクリプションを識別できないわけではないが、上記のコードを変更すると、我々はリスクがあると識別するが、Scalaスクリプトは識別しないサブスクリプションがいくつか見つかるだろう。この理由を調べることは有益かもしれませんが、Scala スクリプトがリスクがあると認識しているのに、私たちが認識していないメンバーがいない限り、このデータセットを使用して、テストデータセットとトレーニングデータセットのサブスクリプションの特徴を導き出すことができるはずです。
# MAGIC 
# MAGIC 最後の数セルで得られたサブスクリプションの期間情報を活用して、アカウントレベルの変更を検出するためにトランザクションログを強化することができます。 この情報は、次のノートブックでトランザクションの特徴を生成するための基礎となります。

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.transactions_enhanced;
# MAGIC 
# MAGIC CREATE TABLE kkbox.transactions_enhanced
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     b.subscription_id,
# MAGIC     a.*,
# MAGIC     COALESCE( DATEDIFF(a.transaction_date, LAG(a.transaction_date, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)), 0) as days_since_last_transaction,
# MAGIC     COALESCE( a.plan_list_price - LAG(a.plan_list_price, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_list_price,
# MAGIC     COALESCE(a.actual_amount_paid - LAG(a.actual_amount_paid, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_actual_amount_paid,
# MAGIC     COALESCE(a.discount - LAG(a.discount, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_discount,
# MAGIC     COALESCE(a.payment_plan_days - LAG(a.payment_plan_days, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date), 0) as change_in_payment_plan_days,
# MAGIC     CASE WHEN (a.payment_method_id != LAG(a.payment_method_id, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)) THEN 1 ELSE 0 END  as change_in_payment_method_id,
# MAGIC     CASE
# MAGIC       WHEN a.is_cancel = LAG(a.is_cancel, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date) THEN 0
# MAGIC       WHEN a.is_cancel = 0 THEN -1
# MAGIC       ELSE 1
# MAGIC       END as change_in_cancellation,
# MAGIC     CASE
# MAGIC       WHEN a.is_auto_renew = LAG(a.is_auto_renew, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date) THEN 0
# MAGIC       WHEN a.is_auto_renew = 0 THEN -1
# MAGIC       ELSE 1
# MAGIC       END as change_in_auto_renew,
# MAGIC     COALESCE( DATEDIFF(a.membership_expire_date, LAG(a.membership_expire_date, 1) OVER(PARTITION BY b.subscription_id ORDER BY a.transaction_date)), 0) as days_change_in_membership_expire_date
# MAGIC 
# MAGIC   FROM kkbox.transactions_clean a
# MAGIC   INNER JOIN kkbox.subscription_windows b
# MAGIC     ON a.msno=b.msno AND 
# MAGIC        a.transaction_date BETWEEN b.subscription_start AND b.subscription_end
# MAGIC   ORDER BY 
# MAGIC     a.msno,
# MAGIC     a.transaction_date;
# MAGIC     
# MAGIC SELECT * FROM kkbox.transactions_enhanced;

# COMMAND ----------

# MAGIC %md ### ステップ4: 日付表の作成
# MAGIC 
# MAGIC 最後に、トランザクションログとユーザーアクティビティデータの両方から、アクティビティのない日を調べて特徴を導き出したいと思うことがあるでしょう。 この分析を容易にするために、データセットの開始日から終了日までの各日付に1つのレコードを含むテーブルを生成するとよいでしょう。 これらのデータは、2015年1月1日から2017年3月31日までのデータであることがわかっています。 それを考慮して、次のようにそのようなテーブルを生成することができます。

# COMMAND ----------

# calculate days in range
start_date = date(2015, 1, 1)
end_date = date(2017, 3, 31)
days = end_date - start_date

# generate temp view of dates in range
( spark
    .range(0, days.days)  
    .withColumn('start_date', lit(start_date.strftime('%Y-%m-%d')))  # first date in activity dataset
    .selectExpr('date_add(start_date, CAST(id as int)) as date')
    .createOrReplaceTempView('dates')
  )

# persist data to SQL table
_ = spark.sql('DROP TABLE IF EXISTS kkbox.dates') 
_ = spark.sql('CREATE TABLE kkbox.dates USING DELTA AS SELECT * FROM dates')

# display SQL table content
display(spark.table('kkbox.dates').orderBy('date'))
