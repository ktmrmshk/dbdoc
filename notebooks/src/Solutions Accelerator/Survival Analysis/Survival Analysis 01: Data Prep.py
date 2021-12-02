# Databricks notebook source
# MAGIC %md 今回のノートでは、サブスクリプションモデルにおける顧客離脱のパターンを理解するために、2つの主要な生存時間分析技術をどのように適用できるかを検討します。 このノートブックでは、分析用に公開されているデータセットを用意します。 このデータを基にして、顧客離脱の理解と予測のための探索的な分析とモデリングを行います。

# COMMAND ----------

# MAGIC %md **注意** このノートは2020年7月20日に改訂されました。

# COMMAND ----------

# MAGIC %md ##ステップ1：生データファイルへのアクセス
# MAGIC 
# MAGIC 今回使用するデータセットは、2018年にKaggleで開催された[KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)のものです。データへのアクセスには、Kaggleへの認証と組織の規約への同意が必要です。 これを済ませると、データをダウンロードして解凍し、お好みのクラウドストレージレイヤーにアップロードすることができます。 このブログでは、この作業がすでに行われており、各ファイルが*/mnt/kkbox/*という名前の[マウントポイント](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs)の下の適切なサブフォルダにアップロードされていることを前提としています。
# MAGIC 
# MAGIC このデータセットには多数のファイルがありますが、ここではトランザクションログとメンバーファイルのみを使用します。他のファイルに含まれる情報は、この分析にとって非常に興味深いものです。しかし、トランザクションログで参照されるサブスクリプションのうち、これらの他のファイルにデータがあるのはほんの一部に過ぎません。 サービス利用状況の全体像を把握できる少数の顧客に分析を限定するのではなく、トランザクション・ログとメンバー・ファイルのより基本的な情報に分析を集中させます。
# MAGIC 
# MAGIC マウントポイント *i.e. */mnt/kkbox/* の下では、transactions.csv と transactions_v2.csv ファイルが *transactions* というサブフォルダに、members_v3.csv が *members* というサブフォルダにアップロードされていることが予想されます。これらのファイルが適切な場所に読み込まれたので、ファイルを[Delta Lake-backed tables](https://databricks.com/product/delta-lake-on-databricks)に読み込んで、後続のステップでより簡単に利用できるようにします。

# COMMAND ----------

# MAGIC %fs mount s3a://databricks-ktmr-s3/kkbox /mnt/kkbox

# COMMAND ----------

# DBTITLE 1,必要なライブラリの読み込み
import shutil
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,古いデータベースオブジェクトの削除
# delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS kkbox CASCADE')

# drop any old delta lake files that might have been created
shutil.rmtree('/dbfs/mnt/kkbox/silver', ignore_errors=True)

# COMMAND ----------

# DBTITLE 1,データを格納するデータベースの作成
# create database to house SQL tables
_ = spark.sql('CREATE DATABASE kkbox')

# COMMAND ----------

# DBTITLE 1,トランザクションデータの準備
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
spark.sql('''
  CREATE TABLE kkbox.transactions
  USING DELTA 
  LOCATION '/mnt/kkbox/silver/transactions'
  ''')

# COMMAND ----------

# DBTITLE 1,Membersデータセットの準備
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
spark.sql('''
  CREATE TABLE kkbox.members 
  USING DELTA 
  LOCATION '/mnt/kkbox/silver/members'
  ''')

# COMMAND ----------

# MAGIC %md ##Step 2: メンバーシップイベントデータセットの構築
# MAGIC 
# MAGIC データがよりアクセスしやすい形式に読み込まれたので、サブスクリプションの寿命情報の構築を開始しましょう。 そのためには、トランザクション・ログに含まれるメンバーシップ関連の情報を調査する必要があります。多くのアプリケーションでよく見られるように、トランザクション・ログ・データには、サブスクリプションの更新以外にも多くの情報が記録されています。 分析に関係のないエントリを除外し、顧客がいつ加入したのか、加入内容が時間の経過とともにどのように変化したのか、そして最終的に顧客がいつ加入を解消したのかを示す加入内容の変化の画像をつなぎ合わせる必要があります（そのようなイベントが発生した場合）。
# MAGIC 
# MAGIC データセットの制限や、Kaggleチャレンジのドキュメントに記載されているKKBoxのビジネスモデルの特徴を考慮すると、このプロセスはかなり複雑です。 必要なロジックを開発するために、1人の顧客、*msno = WAMleIuG124oDSSdhcZIECwLcHHbk4or0y6gxkK7t8=*に焦点を絞り、この人物に関連するトランザクションログ情報を調べてみましょう。

# COMMAND ----------

# DBTITLE 1,トランザクション履歴
# MAGIC %sql  -- all transactions for customer msno WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8=
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.transactions a
# MAGIC WHERE a.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY a.transaction_date, a.membership_expire_date

# COMMAND ----------

# MAGIC %md トランザクションログは非常に忙しく、特定のトランザクション日(*transaction_date*)に複数のエントリが記録されていることが多く、このお客様の場合、以下のようないくつかの日付で観察できます。
# MAGIC 2016-09-25. 文書や、このデータセットを使用した他の人の分析に基づいて、支払いプランの日数(*payment_plan_days*)の値が0以下のレコードは、分析には関係ないと考えることができます。 これでも、ある取引日のログにはたくさんのエントリが残っています。
# MAGIC 
# MAGIC ある日付のトランザクションエントリをもう少し詳しく見てみると、多くはサブスクリプションの有効期限(*membership_expire_date*)を変更しているようです。 トランザクションの日付フィールドの時間部分が切り捨てられているため、どのエントリーが特定の日付の*最終*エントリーなのかを判断する方法はありません。このような制限があるため、ここでは、ある取引日の最も遠い未来の有効期限が、その時点でのアカウントの有効期限であると仮定します。 これは完全に有効な仮定ではないかもしれませんが、サブスクリプションの変更イベントは、後の取引日にフォローアップのトランザクションを引き起こし、これらに関連する有効期限はより安定しているように思われます。 つまり、ある取引日に正確な情報が得られなくても、アカウントに関連するエントリーの範囲を見ることで、有効期限を正確に理解することができるのです。

# COMMAND ----------

# DBTITLE 1,トランザクション履歴の要約
# MAGIC %sql  -- drop payment_plan_days of zero or less
# MAGIC       --   and select largest exp date on a trans date  
# MAGIC 
# MAGIC SELECT 
# MAGIC   a.msno, 
# MAGIC   a.transaction_date as trans_at, 
# MAGIC   MAX(a.membership_expire_date) as expires_at 
# MAGIC FROM kkbox.transactions a
# MAGIC WHERE  
# MAGIC   a.payment_plan_days > 0 AND 
# MAGIC   a.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC GROUP BY
# MAGIC   a.msno, a.transaction_date
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md このお客様のより詳細な取引履歴を確認すると、2015年12月24日に発生したデータに別の奇妙な現象が見られます（*trans_at = 2015-12-24*）。この日、アカウントの有効期限は、サブスクリプション開始前に発生する日付にリセットされています。 次の取引日には、アカウントは未来の日付にリセットされています。
# MAGIC 
# MAGIC 過去に遡って記録されるのは、自動更新ステータスの変更など、何らかのサブスクリプション管理活動が原因と考えられます。 理由が何であれ、過去にさかのぼって記録された値を解約識別の目的で考慮すべきではないことは明らかです。 代わりに、過去に遡った有効期限がログに表示された場合、それが関連付けられているトランザクションの日付にリセットされるようなロジックを追加することができます。

# COMMAND ----------

# DBTITLE 1,トランザクション履歴の修正
# MAGIC %sql  -- correct expiration dates that have been backdated
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno,
# MAGIC   x.trans_at,
# MAGIC   CASE   -- if expiration date is prior to transaction date, then expiration date = transaction date
# MAGIC     WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC     ELSE expires_at 
# MAGIC     END as expires_at
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.msno, 
# MAGIC     a.transaction_date as trans_at, 
# MAGIC     MAX(a.membership_expire_date) as expires_at 
# MAGIC   FROM kkbox.transactions a
# MAGIC   WHERE  
# MAGIC     a.payment_plan_days > 0 
# MAGIC   GROUP BY
# MAGIC     a.msno, a.transaction_date
# MAGIC   ) x
# MAGIC WHERE x.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md 最後に、全体の取引データセットが2017年3月31日に終了することに注意しなければなりません。 この点に関して、我々は2017年4月1日の*有効な現在の日付*から逆算して取引履歴を観察しています。
# MAGIC 
# MAGIC このデータセットに関連するKaggleチャレンジのドキュメントで、KKBoxは、サブスクリプションの有効期限から30日が経過し、有効期限を前倒しするようなアカウントへの変更がない限り、顧客が解約したと考えるべきではないと説明しています。 この点を考慮して、2017年4月1日のトランザクションデータセットにダミーのエントリを入れて、30日間の有効期限がまだ完了していないアカウントを解約したとみなさないようにする必要があります。 このエントリの効果を完全に理解するには、このノートブックの次のセクションで解約を特定する方法を見てみる必要があります。

# COMMAND ----------

# DBTITLE 1,ダミー取引の追加（2017年4月1日分）
# MAGIC %sql  -- add dummy transaction entries for 2017-04-01 to prevent misclassification of churn in subsequent steps 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno,
# MAGIC   x.trans_at,
# MAGIC   CASE 
# MAGIC     WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC     ELSE expires_at 
# MAGIC     END as expires_at
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.msno, 
# MAGIC     a.transaction_date as trans_at, 
# MAGIC     MAX(a.membership_expire_date) as expires_at 
# MAGIC   FROM kkbox.transactions a
# MAGIC   WHERE  
# MAGIC     a.payment_plan_days > 0
# MAGIC   GROUP BY
# MAGIC     a.msno, a.transaction_date
# MAGIC   UNION ALL
# MAGIC   SELECT DISTINCT  -- dummy entries to protect churn calculations
# MAGIC     msno,
# MAGIC     TO_DATE('2017-04-01') as trans_at,
# MAGIC     TO_DATE('2017-04-01') as expires_at
# MAGIC   FROM kkbox.transactions
# MAGIC   ) x
# MAGIC WHERE x.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md ##ステップ3：解約イベントの特定
# MAGIC 
# MAGIC これで、このお客様の会員関連イベントの履歴が比較的きれいに残りました。これにより、各トランザクションを直前のトランザクションと比較して、新しい有効期限が設定されたかどうかを確認することができます。 このようなイベントは、サブスクリプションの更新や延長を表している可能性があります。

# COMMAND ----------

# DBTITLE 1,Identify Membership-Meaningful Events 
# MAGIC %sql  -- compare membership-relevant transactions to prior transactions to identify expiration changes
# MAGIC 
# MAGIC WITH trans AS (  -- membership event dataset (from previous section)
# MAGIC   SELECT  -- -----------------------------------------------
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT 
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      -- -----------------------------------------------
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     CASE   -- idnetify meaningful events that change expiration date
# MAGIC       WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  -- new customer registration
# MAGIC       WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 -- change in expiration date
# MAGIC       ELSE 0
# MAGIC       END as meaningful_event
# MAGIC   FROM trans
# MAGIC   WHERE msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md 意義のある会員イベントだけに焦点を絞ると、前回の有効期限と今回の取引日（有効期限が前倒しされた場合）の間の日数を調べることができます。 有効期限から30日以上経過しても、意味のある取引（例：有効期限を前倒しする取引）がない場合、解約が発生したと考えられます。 
# MAGIC 
# MAGIC もちろん、この計算は、有効期限後にトランザクションが記録されていることが前提となります。 多くのアカウントでは、お客様が6ヶ月から1年の間離れていて、その後アカウントを更新しているように見えます。 このようなケースでは、解約イベントが記録され、その後のアクティベーションによって新たな会員イベントの連鎖が形成される可能性があります。 後続のトランザクションが発生しない状況では、2017 年 4 月 1 日に挿入されたダミーレコードによって、解約イベントが発生したことを正しく認識することができます。

# COMMAND ----------

# DBTITLE 1,Churned Flagの追加
# MAGIC %sql  -- identify churn events as those there the next transaction is 30+ days from prior expiration date
# MAGIC 
# MAGIC WITH trans AS (  
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC   
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     CASE  -- identify churn events
# MAGIC       WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC       ELSE 0
# MAGIC       END as churn_event
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE 
# MAGIC         WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  
# MAGIC         WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 
# MAGIC         ELSE 0
# MAGIC         END as meaningful_event
# MAGIC     FROM trans
# MAGIC     )
# MAGIC   WHERE 
# MAGIC     meaningful_event=1
# MAGIC     AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md 結果を見ると、下流のトランザクションが、再申し込みであれ、単なるダミーのトランザクションであれ、解約に関連するものとしてフラグが立てられていることに気づきます。 フラグを1つのトランザクションの日付に戻すことで、フラグが、解約が発生したと直感的に理解できるイベントに関連付けられます。

# COMMAND ----------

# DBTITLE 1,Churnフラグを適切に配置
# MAGIC %sql -- move churn flag to prior transaction record
# MAGIC 
# MAGIC WITH trans AS ( 
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned -- adjust churn flag assignment
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE
# MAGIC         WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC         ELSE 0
# MAGIC         END as churn_event
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1
# MAGIC           ELSE 0
# MAGIC           END as meaningful_event
# MAGIC       FROM trans
# MAGIC       )
# MAGIC     WHERE meaningful_event=1
# MAGIC     )
# MAGIC   WHERE msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md これにより、ダミーのトランザクション・レコードにはNULLフラグの値が残ります。 これらを削除することで、顧客の正確な購読履歴ログが得られます。

# COMMAND ----------

# DBTITLE 1,ダミートランザクションの削除
# MAGIC %sql  -- remove the dummy transactions added earlier
# MAGIC 
# MAGIC WITH trans AS (  
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned 
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE
# MAGIC         WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC         ELSE 0
# MAGIC         END as churn_event
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 
# MAGIC           ELSE 0
# MAGIC           END as meaningful_event
# MAGIC       FROM trans
# MAGIC       )
# MAGIC     WHERE meaningful_event=1
# MAGIC     )
# MAGIC   )
# MAGIC WHERE churned IS NOT NULL  -- remove dummy transaction entries
# MAGIC       AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md 顧客固有の制約を取り除き、最後のSELECTリストを整理し、クエリをビューに変換して、すべての顧客のサブスクリプションでの使用を少しでも容易にしましょう。

# COMMAND ----------

# DBTITLE 1,ビューに変換して再利用
# MAGIC %sql  -- create view from query (with customer-filter removed)
# MAGIC 
# MAGIC DROP VIEW IF EXISTS kkbox.membership_history;
# MAGIC 
# MAGIC CREATE VIEW kkbox.membership_history
# MAGIC AS
# MAGIC   WITH trans AS (  
# MAGIC     SELECT  
# MAGIC       x.msno,
# MAGIC       x.trans_at,
# MAGIC       CASE 
# MAGIC         WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC         ELSE expires_at 
# MAGIC         END as expires_at
# MAGIC     FROM (
# MAGIC       SELECT 
# MAGIC         a.msno, 
# MAGIC         a.transaction_date as trans_at, 
# MAGIC         MAX(a.membership_expire_date) as expires_at 
# MAGIC       FROM kkbox.transactions a
# MAGIC       WHERE  
# MAGIC         a.payment_plan_days > 0
# MAGIC       GROUP BY
# MAGIC         a.msno, a.transaction_date
# MAGIC       UNION ALL
# MAGIC       SELECT DISTINCT  
# MAGIC         msno,
# MAGIC         TO_DATE('2017-04-01') as trans_at,
# MAGIC         TO_DATE('2017-04-01') as expires_at
# MAGIC       FROM kkbox.transactions
# MAGIC       ) x
# MAGIC     )      
# MAGIC SELECT 
# MAGIC   msno,
# MAGIC   trans_at,
# MAGIC   expires_at,
# MAGIC   churned
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned 
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE
# MAGIC           WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC           ELSE 0
# MAGIC           END as churn_event
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           msno,
# MAGIC           LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC           trans_at,
# MAGIC           expires_at,
# MAGIC           CASE 
# MAGIC             WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1
# MAGIC             WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1
# MAGIC             ELSE 0
# MAGIC             END as meaningful_event
# MAGIC         FROM trans
# MAGIC         )
# MAGIC       WHERE meaningful_event=1
# MAGIC       )
# MAGIC     )
# MAGIC WHERE churned IS NOT NULL 
# MAGIC       --AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md すべてのお客様のサブスクリプションの有効化、期限切れ、更新の履歴を調べることができるようになりました。

# COMMAND ----------

# DBTITLE 1,全会員のメンバーシップ変更履歴
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.membership_history

# COMMAND ----------

# MAGIC %md ##ステップ4: 購読情報の抽出
# MAGIC 
# MAGIC メンバーシップの履歴が集まり、解約イベントのフラグが立ったので、サブスクリプションのデータセットを作成します。 このデータセットには、各サブスクリプションについて顧客ごとに1つのエントリが必要です。 サブスクリプションは、特定の取引日以降に発生した最も古い解約イベントで終了したと識別されます。 また、その日に関連する最も古い取引で開始されたと識別される。1人の顧客（*msno*値で識別される）は、複数のサブスクリプションを持つことができるが、ある時点でアクティブなのは1つだけである。

# COMMAND ----------

# DBTITLE 1,サブスクリプション履歴のview
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.subscriptions;
# MAGIC 
# MAGIC CREATE TABLE kkbox.subscriptions
# MAGIC USING delta
# MAGIC AS
# MAGIC   SELECT
# MAGIC     ROW_NUMBER() OVER(ORDER BY p.starts_at) as subscription_id,
# MAGIC     p.msno,
# MAGIC     p.starts_at,
# MAGIC     p.ends_at,
# MAGIC     1 + DATEDIFF(COALESCE(p.ends_at,'2017-04-01'), p.starts_at) as duration_days,
# MAGIC     CASE WHEN ends_at IS NULL THEN 0 ELSE 1 END as churned,
# MAGIC     COUNT(*) OVER(PARTITION BY p.msno ORDER BY p.starts_at) - 1 as prior_subscriptions
# MAGIC   FROM (  -- first transaction prior to or on churn date
# MAGIC     SELECT 
# MAGIC       x.msno,
# MAGIC       MIN(x.trans_at) as starts_at,
# MAGIC       x.churns_at as ends_at
# MAGIC     FROM (  -- membership history with associated churn dates (for each transaction)
# MAGIC       SELECT  
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         a.expires_at,
# MAGIC         a.churned,
# MAGIC         MIN(b.expires_at) as churns_at
# MAGIC       FROM kkbox.membership_history a  -- all events 
# MAGIC       LEFT OUTER JOIN kkbox.membership_history b  -- churn events only
# MAGIC         ON a.msno = b.msno AND  -- match customer
# MAGIC            1 = b.churned AND    -- the right-hand (b) dataset is limited to churn events
# MAGIC            a.trans_at <= b.expires_at -- transactions prior to or on date of churn
# MAGIC       GROUP BY 
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         a.expires_at,
# MAGIC         a.churned
# MAGIC         ) x
# MAGIC     GROUP BY
# MAGIC       x.msno,
# MAGIC       x.churns_at
# MAGIC     ) p
# MAGIC   ORDER BY p.msno, starts_at, ends_at

# COMMAND ----------

# DBTITLE 1,サブスクリプション履歴
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.subscriptions
# MAGIC ORDER BY subscription_id
