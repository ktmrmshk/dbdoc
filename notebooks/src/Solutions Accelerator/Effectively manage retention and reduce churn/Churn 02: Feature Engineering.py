# Databricks notebook source
# MAGIC %md このノートブックの目的は、トランザクションログとユーザーログから、解約を予測するために使用される限られた数の特徴をエンジニアリングすることです。 このノートブックは、**Databricks ML 7.1+**と**CPUベース**のノードを利用したクラスタ上で実行する必要があります。

# COMMAND ----------

# MAGIC %md ###ステップ1: トランザクションログから特徴を抽出する
# MAGIC 
# MAGIC トランザクションログのデータにアクセスする際に重要なのは、調査対象期間の開始直前にアクセスできる情報に限定し、開始後にはアクセスしないことです。 例えば、2017年2月の解約を調査する場合、2017年1月31日までのトランザクションログデータを調査しますが、2017年2月1日以降は調査しません。
# MAGIC 
# MAGIC 対象期間に向けて実行可能なサブスクリプションと、それらのサブスクリプションがいつ開始されたかを知ることで、サブスクリプションの開始日から対象期間の開始日の前日までの日付の範囲を定義し、そこからトランザクションログの特徴を導き出すことができます。 これらの範囲は次のセルで計算されますが、ここでは、その下の機能エンジニアリングクエリに適用される前にロジックを簡単に確認できるように、単独で表示されています。

# COMMAND ----------

# DBTITLE 1,トレーニング期間のトランザクションログWindows（2017年2月）
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.msno,
# MAGIC   b.subscription_id,
# MAGIC   b.subscription_start as start_at,
# MAGIC   c.last_at
# MAGIC FROM kkbox.train a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TRAINING PERIOD
# MAGIC LEFT OUTER JOIN (   -- subscriptions not yet churned heading into the period of interest
# MAGIC   SELECT *
# MAGIC   FROM kkbox.subscription_windows 
# MAGIC   WHERE subscription_start < '2017-02-01' AND subscription_end > DATE_ADD('2017-02-01', -30)
# MAGIC   )b
# MAGIC   ON a.msno=b.msno
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT            -- last transaction date prior to the start of the at-risk period (we could also have just set this to the day prior to the start of the period of interest)
# MAGIC     subscription_id,
# MAGIC     MAX(transaction_date) as last_at
# MAGIC   FROM kkbox.transactions_enhanced
# MAGIC   WHERE transaction_date < '2017-02-01'
# MAGIC   GROUP BY subscription_id
# MAGIC   ) c
# MAGIC   ON b.subscription_id=c.subscription_id

# COMMAND ----------

# MAGIC %md これらの日付範囲を使用して、*現在の*サブスクリプションのトランザクションログから機能を導出することができます。 なお、サブスクライバーの過去のすべてのサブスクリプションから情報を抽出することも可能ですが、今回の演習では、現在のサブスクリプションに関連する情報と過去のサブスクリプションの単純なカウントに機能工学を限定しています。

# COMMAND ----------

# DBTITLE 1,トレーニング期間中のトランザクションログ特徴量（2017年2月）
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.train_trans_features;
# MAGIC 
# MAGIC CREATE TABLE kkbox.train_trans_features
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH transaction_window (  -- this is the query from above defined as a CTE
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       b.subscription_start as start_at,
# MAGIC       c.last_at
# MAGIC     FROM kkbox.train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM kkbox.subscription_windows 
# MAGIC       WHERE subscription_start < '2017-02-01' AND subscription_end > DATE_ADD('2017-02-01', -30)
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT  
# MAGIC         subscription_id,
# MAGIC         MAX(transaction_date) as last_at
# MAGIC       FROM kkbox.transactions_enhanced
# MAGIC       WHERE transaction_date < '2017-02-01'
# MAGIC       GROUP BY subscription_id
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC     a.msno,
# MAGIC     YEAR(b.start_at) as start_year,
# MAGIC     MONTH(b.start_at) as start_month,
# MAGIC     DATEDIFF(b.last_at, b.start_at) as subscription_age,
# MAGIC     c.renewals,
# MAGIC     c.total_list_price,
# MAGIC     c.total_amount_paid,
# MAGIC     c.total_discount,
# MAGIC     DATEDIFF('2017-02-01', LAST(a.transaction_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_since_last_account_action,
# MAGIC     LAST(a.plan_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_plan_list_price,
# MAGIC     LAST(a.actual_amount_paid) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_actual_amount_paid,
# MAGIC     LAST(a.discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_discount,
# MAGIC     LAST(a.payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_plan_days,
# MAGIC     LAST(a.payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_method,
# MAGIC     LAST(a.is_cancel) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_cancel,
# MAGIC     LAST(a.is_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_auto_renew,
# MAGIC     LAST(a.change_in_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_list_price,
# MAGIC     LAST(a.change_in_discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_discount,
# MAGIC     LAST(a.change_in_payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_plan_days,
# MAGIC     LAST(a.change_in_payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_method_id,
# MAGIC     LAST(a.change_in_cancellation) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_cancellation,
# MAGIC     LAST(a.change_in_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_auto_renew,
# MAGIC     LAST(a.days_change_in_membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_days_change_in_membership_expire_date,
# MAGIC     DATEDIFF('2017-02-01', LAST(a.membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_until_expiration,
# MAGIC     d.total_subscription_count,
# MAGIC     e.city,
# MAGIC     CASE WHEN e.bd < 10 THEN NULL WHEN e.bd > 70 THEN NULL ELSE e.bd END as bd,
# MAGIC     CASE WHEN LOWER(e.gender)='female' THEN 0 WHEN LOWER(e.gender)='male' THEN 1 ELSE NULL END as gender,
# MAGIC     e.registered_via  
# MAGIC   FROM kkbox.transactions_enhanced a
# MAGIC   INNER JOIN transaction_window b
# MAGIC     ON a.subscription_id=b.subscription_id AND a.transaction_date = b.last_at
# MAGIC   INNER JOIN (
# MAGIC     SELECT  -- summary stats for current subscription
# MAGIC       x.subscription_id,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.plan_list_price ELSE 0 END) as total_list_price,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.actual_amount_paid ELSE 0 END) as total_amount_paid,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.discount ELSE 0 END) as total_discount,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN 1 ELSE 0 END) as renewals
# MAGIC     FROM kkbox.transactions_enhanced x
# MAGIC     INNER JOIN transaction_window y
# MAGIC       ON x.subscription_id=y.subscription_id AND x.transaction_date BETWEEN y.start_at AND y.last_at
# MAGIC     GROUP BY x.subscription_id
# MAGIC     ) c
# MAGIC     ON a.subscription_id=c.subscription_id
# MAGIC   INNER JOIN (
# MAGIC     SELECT  -- count of all unique subscriptions for each customer
# MAGIC       msno,
# MAGIC       COUNT(*) as total_subscription_count
# MAGIC     FROM kkbox.subscription_windows
# MAGIC     WHERE subscription_start < '2017-02-01'
# MAGIC     GROUP BY msno
# MAGIC     ) d
# MAGIC     ON a.msno=d.msno
# MAGIC   LEFT OUTER JOIN kkbox.members e
# MAGIC     ON a.msno=e.msno;
# MAGIC     
# MAGIC SELECT * FROM kkbox.train_trans_features;

# COMMAND ----------

# MAGIC %md 日付を変更すると、テスト期間である2017年3月にも同様の機能を導き出すことができます。

# COMMAND ----------

# DBTITLE 1,テスト期間中のトランザクションログ特徴量（2017年3月）
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.test_trans_features;
# MAGIC 
# MAGIC CREATE TABLE kkbox.test_trans_features
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   WITH transaction_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       b.subscription_start as start_at,
# MAGIC       c.last_at
# MAGIC     FROM kkbox.test a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TESTING PERIOD
# MAGIC     LEFT OUTER JOIN (  -- subscriptions not yet churned heading into the period of interest
# MAGIC       SELECT *
# MAGIC       FROM kkbox.subscription_windows 
# MAGIC       WHERE subscription_start < '2017-03-01' AND subscription_end > DATE_ADD('2017-03-01', -30) 
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT  
# MAGIC         subscription_id,
# MAGIC         MAX(transaction_date) as last_at
# MAGIC       FROM kkbox.transactions_enhanced
# MAGIC       WHERE transaction_date < '2017-03-01'
# MAGIC       GROUP BY subscription_id
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id
# MAGIC       )
# MAGIC   SELECT
# MAGIC     a.msno,
# MAGIC     YEAR(b.start_at) as start_year,
# MAGIC     MONTH(b.start_at) as start_month,
# MAGIC     DATEDIFF(b.last_at, b.start_at) as subscription_age,
# MAGIC     c.renewals,
# MAGIC     c.total_list_price,
# MAGIC     c.total_amount_paid,
# MAGIC     c.total_discount,
# MAGIC     DATEDIFF('2017-03-01', LAST(a.transaction_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_since_last_account_action,
# MAGIC     LAST(a.plan_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_plan_list_price,
# MAGIC     LAST(a.actual_amount_paid) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_actual_amount_paid,
# MAGIC     LAST(a.discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_discount,
# MAGIC     LAST(a.payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_plan_days,
# MAGIC     LAST(a.payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_payment_method,
# MAGIC     LAST(a.is_cancel) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_cancel,
# MAGIC     LAST(a.is_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_is_auto_renew,
# MAGIC     LAST(a.change_in_list_price) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_list_price,
# MAGIC     LAST(a.change_in_discount) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_discount,
# MAGIC     LAST(a.change_in_payment_plan_days) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_plan_days,
# MAGIC     LAST(a.change_in_payment_method_id) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_payment_method_id,
# MAGIC     LAST(a.change_in_cancellation) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_cancellation,
# MAGIC     LAST(a.change_in_auto_renew) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_change_in_auto_renew,
# MAGIC     LAST(a.days_change_in_membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date) as last_days_change_in_membership_expire_date,
# MAGIC     DATEDIFF('2017-03-01', LAST(a.membership_expire_date) OVER(PARTITION BY a.subscription_id ORDER BY a.transaction_date)) as days_until_expiration,
# MAGIC     d.total_subscription_count,
# MAGIC     e.city,
# MAGIC     CASE WHEN e.bd < 10 THEN NULL WHEN e.bd > 70 THEN NULL ELSE e.bd END as bd,
# MAGIC     CASE WHEN LOWER(e.gender)='female' THEN 0 WHEN LOWER(e.gender)='male' THEN 1 ELSE NULL END as gender,
# MAGIC     e.registered_via  
# MAGIC   FROM kkbox.transactions_enhanced a
# MAGIC   INNER JOIN transaction_window b
# MAGIC     ON a.subscription_id=b.subscription_id AND a.transaction_date = b.last_at
# MAGIC   INNER JOIN (
# MAGIC     SELECT 
# MAGIC       x.subscription_id,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.plan_list_price ELSE 0 END) as total_list_price,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.actual_amount_paid ELSE 0 END) as total_amount_paid,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN x.discount ELSE 0 END) as total_discount,
# MAGIC       SUM(CASE WHEN x.days_change_in_membership_expire_date > 0 THEN 1 ELSE 0 END) as renewals
# MAGIC     FROM kkbox.transactions_enhanced x
# MAGIC     INNER JOIN transaction_window y
# MAGIC       ON x.subscription_id=y.subscription_id AND x.transaction_date BETWEEN y.start_at AND y.last_at
# MAGIC     GROUP BY x.subscription_id
# MAGIC     ) c
# MAGIC     ON a.subscription_id=c.subscription_id
# MAGIC   INNER JOIN (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       COUNT(*) as total_subscription_count
# MAGIC     FROM kkbox.subscription_windows
# MAGIC     WHERE subscription_start < '2017-03-01'
# MAGIC     GROUP BY msno
# MAGIC     ) d
# MAGIC     ON a.msno=d.msno
# MAGIC   LEFT OUTER JOIN kkbox.members e
# MAGIC     ON a.msno=e.msno;
# MAGIC     
# MAGIC SELECT * FROM kkbox.test_trans_features;

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 上記のトランザクション機能を検討することで、さらに多くの機能を導き出すことができることに気づくかもしれません。 この演習の目的は、特徴の種類を網羅的に検討することではなく、モデルを訓練するための潜在的な特徴の有意義なサブセットを生成することです。
# MAGIC 
# MAGIC 先に進む前に、トレーニング期間とテスト期間のデータセットに含まれるすべての顧客の特徴を確認しましょう。 これらのクエリでは、マッチしないレコードが0件になるようにします。

# COMMAND ----------

# DBTITLE 1,Features for All Training Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM kkbox.train a
# MAGIC LEFT OUTER JOIN kkbox.train_trans_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# DBTITLE 1,Features for All Testing Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM kkbox.test a
# MAGIC LEFT OUTER JOIN kkbox.test_trans_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# MAGIC %md ###ステップ2：ユーザーログから機能を抽出する
# MAGIC 
# MAGIC トランザクションログの機能と同様に、現在のリスクのあるサブスクリプションのユーザーアクティビティを調査するために、日付の範囲を定義する必要があります。 このロジックは、先ほどのロジックとは異なり、KKBoxではユーザーが有効期限後30日間サブスクリプションを使用し続けることができるため、対象期間に入ったすべてのユーザーアクティビティを考慮します。 期限切れのサブスクリプションがまだ使用されていることを知ることは、解約の意図を示す重要な指標となるはずです。 
# MAGIC 
# MAGIC さらに、ユーザーログからの特徴抽出は、対象期間の開始前30日以内に発生したアクティビティに限定していることにも留意する必要があります。トランザクションログと同様に、契約開始時の使用状況、契約期間中（対象期間開始前）の使用状況、対象期間に入るまでの期間の違いなど、さらに多くの特徴を導き出すことができます。 このように機能を限定することは任意です。なぜなら、この演習の目的は機能の網羅的なセットを作成することではなく、モデルのトレーニングに使用できる意味のあるセットを作成することだからです。
# MAGIC 
# MAGIC 以上を踏まえて、ユーザーログから特徴量を抽出するための日付範囲を計算してみましょう。

# COMMAND ----------

# DBTITLE 1,トレーニング期間中のユーザーログWindows（2017年2月）
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   a.msno,
# MAGIC   b.subscription_id,
# MAGIC   CASE 
# MAGIC     WHEN b.subscription_start < DATE_ADD('2017-02-01', -30) THEN DATE_ADD('2017-02-01', -30) -- cap subscription info to 30-days prior to start of period
# MAGIC     ELSE b.subscription_start 
# MAGIC     END as start_at,
# MAGIC   DATE_ADD('2017-02-01', -1) as end_at,
# MAGIC   c.last_at as last_exp_at
# MAGIC FROM kkbox.train a  -- LIMIT ANALYSIS TO AT-RISK SUBSCRIBERS IN THE TRAINING PERIOD
# MAGIC LEFT OUTER JOIN (   -- subscriptions not yet churned heading into the period of interest 
# MAGIC   SELECT *
# MAGIC   FROM kkbox.subscription_windows 
# MAGIC   WHERE subscription_start < '2017-02-01' AND subscription_end > DATE_ADD('2017-02-01', -30)
# MAGIC   )b
# MAGIC   ON a.msno=b.msno
# MAGIC LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC   SELECT
# MAGIC     x.subscription_id,
# MAGIC     y.membership_expire_date as last_at
# MAGIC   FROM (
# MAGIC     SELECT  -- last subscription transaction before start of this period
# MAGIC       subscription_id,
# MAGIC       MAX(transaction_date) as transaction_date
# MAGIC     FROM kkbox.transactions_enhanced
# MAGIC     WHERE transaction_date < '2017-02-01'
# MAGIC     GROUP BY subscription_id
# MAGIC     ) x
# MAGIC   INNER JOIN kkbox.transactions_enhanced y
# MAGIC     ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC   ) c
# MAGIC   ON b.subscription_id=c.subscription_id  

# COMMAND ----------

# MAGIC %md これらの日付範囲を使用して、ユーザーログの分析を制限することができます。 注意すべき点は、ユーザーが特定の日に複数のストリーミングセッションを行う可能性があることです。 そのため、ユーザーログを簡単に利用できるようにするために、日レベルの統計を導き出したいと思います。 さらに、日レベルの統計値を日付範囲のデータセット（前回のノートで作成した「kkbox.dates」）と結合することで、対象範囲の各日について1つのレコードを得ることができます。 活動と非活動のパターンを理解することは、どの購読者が解約するかを判断するのに役立つでしょう。

# COMMAND ----------

# DBTITLE 1,日単位のユーザーアクティビティ統計を算出（2017年2月）
# MAGIC %sql
# MAGIC 
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start < DATE_ADD('2017-02-01', -30) THEN DATE_ADD('2017-02-01', -30) 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC       DATE_ADD('2017-02-01', -1) as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM kkbox.train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM kkbox.subscription_windows 
# MAGIC       WHERE subscription_start < '2017-02-01' AND subscription_end > DATE_ADD('2017-02-01', -30)
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM kkbox.transactions_enhanced
# MAGIC         WHERE transaction_date < '2017-02-01'
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN kkbox.transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC     )    
# MAGIC SELECT
# MAGIC   a.subscription_id,
# MAGIC   a.msno,
# MAGIC   b.date,
# MAGIC   CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC   CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC   COALESCE(c.session_count, 0) as sessions_total,
# MAGIC   COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC   COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC   COALESCE(c.num_total,0) as number_total
# MAGIC FROM activity_window a
# MAGIC INNER JOIN kkbox.dates b
# MAGIC   ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC LEFT OUTER JOIN (
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     date,
# MAGIC     COUNT(*) as session_count,
# MAGIC     SUM(total_secs) as total_secs,
# MAGIC     SUM(num_uniq) as num_uniq,
# MAGIC     SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC   FROM kkbox.user_logs
# MAGIC   GROUP BY msno, date
# MAGIC   ) c
# MAGIC   ON a.msno=c.msno AND b.date=c.date
# MAGIC ORDER BY subscription_id, date

# COMMAND ----------

# MAGIC %md 毎日の活動記録が構築されたので、ユーザーアクティビティ機能を形成する要約統計を作成することができます。

# COMMAND ----------

# DBTITLE 1,トレーニング期間（2017年2月）のユーザーアクティビティログの特徴
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.train_act_features;
# MAGIC 
# MAGIC CREATE TABLE kkbox.train_act_features
# MAGIC USING DELTA 
# MAGIC AS
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start < DATE_ADD('2017-02-01', -30) THEN DATE_ADD('2017-02-01', -30) 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC       DATE_ADD('2017-02-01', -1) as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM kkbox.train a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM kkbox.subscription_windows 
# MAGIC       WHERE subscription_start < '2017-02-01' AND subscription_end > DATE_ADD('2017-02-01', -30)
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM kkbox.transactions_enhanced
# MAGIC         WHERE transaction_date < '2017-02-01'
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN kkbox.transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC       ),
# MAGIC   activity (
# MAGIC     SELECT
# MAGIC       a.subscription_id,
# MAGIC       a.msno,
# MAGIC       b.date,
# MAGIC       CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC       CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC       COALESCE(c.session_count, 0) as sessions_total,
# MAGIC       COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC       COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC       COALESCE(c.num_total,0) as number_total
# MAGIC     FROM activity_window a
# MAGIC     INNER JOIN kkbox.dates b
# MAGIC       ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         date,
# MAGIC         COUNT(*) as session_count,
# MAGIC         SUM(total_secs) as total_secs,
# MAGIC         SUM(num_uniq) as num_uniq,
# MAGIC         SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC       FROM kkbox.user_logs
# MAGIC       GROUP BY msno, date
# MAGIC       ) c
# MAGIC       ON a.msno=c.msno AND b.date=c.date
# MAGIC     )
# MAGIC   
# MAGIC SELECT 
# MAGIC   subscription_id,
# MAGIC   msno,
# MAGIC   COUNT(*) as days_total,
# MAGIC   SUM(had_session) as days_with_session,
# MAGIC   COALESCE(SUM(had_session)/COUNT(*),0) as ratio_days_with_session_to_days,
# MAGIC   SUM(after_exp) as days_after_exp,
# MAGIC   SUM(had_session * after_exp) as days_after_exp_with_session,
# MAGIC   COALESCE(SUM(had_session * after_exp)/SUM(after_exp),0) as ratio_days_after_exp_with_session_to_days_after_exp,
# MAGIC   SUM(sessions_total) as sessions_total,
# MAGIC   COALESCE(SUM(sessions_total)/COUNT(*),0) as ratio_sessions_total_to_days_total,
# MAGIC   COALESCE(SUM(sessions_total)/SUM(had_session),0) as ratio_sessions_total_to_days_with_session,
# MAGIC   SUM(sessions_total * after_exp) as sessions_total_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(had_session * after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(seconds_total) as seconds_total,
# MAGIC   COALESCE(SUM(seconds_total)/COUNT(*),0) as ratio_seconds_total_to_days_total,
# MAGIC   COALESCE(SUM(seconds_total)/SUM(had_session),0) as ratio_seconds_total_to_days_with_session,
# MAGIC   SUM(seconds_total * after_exp) as seconds_total_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(had_session * after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_uniq) as number_uniq,
# MAGIC   COALESCE(SUM(number_uniq)/COUNT(*),0) as ratio_number_uniq_to_days_total,
# MAGIC   COALESCE(SUM(number_uniq)/SUM(had_session),0) as ratio_number_uniq_to_days_with_session,
# MAGIC   SUM(number_uniq * after_exp) as number_uniq_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(had_session * after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_total) as number_total,
# MAGIC   COALESCE(SUM(number_total)/COUNT(*),0) as ratio_number_total_to_days_total,
# MAGIC   COALESCE(SUM(number_total)/SUM(had_session),0) as ratio_number_total_to_days_with_session,
# MAGIC   SUM(number_total * after_exp) as number_total_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(after_exp),0) as ratio_number_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(had_session * after_exp),0) as ratio_number_total_after_exp_to_days_after_exp_with_session
# MAGIC FROM activity
# MAGIC GROUP BY subscription_id, msno
# MAGIC ORDER BY msno;
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.train_act_features;

# COMMAND ----------

# MAGIC %md 同じロジックでテスト期間中の機能を生成することができます。

# COMMAND ----------

# DBTITLE 1,テスト期間中のユーザーアクティビティログ特徴量（2017年3月）
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.test_act_features;
# MAGIC 
# MAGIC CREATE TABLE kkbox.test_act_features
# MAGIC USING DELTA 
# MAGIC AS
# MAGIC WITH activity_window (
# MAGIC     SELECT
# MAGIC       a.msno,
# MAGIC       b.subscription_id,
# MAGIC       CASE 
# MAGIC         WHEN b.subscription_start < DATE_ADD('2017-03-01', -30) THEN DATE_ADD('2017-03-01', -30) 
# MAGIC         ELSE b.subscription_start 
# MAGIC         END as start_at,
# MAGIC       DATE_ADD('2017-03-01', -1) as end_at,
# MAGIC       c.last_at as last_exp_at
# MAGIC     FROM kkbox.test a
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT *
# MAGIC       FROM kkbox.subscription_windows 
# MAGIC       WHERE subscription_start < '2017-03-01' AND subscription_end > DATE_ADD('2017-03-01', -30)
# MAGIC       )b
# MAGIC       ON a.msno=b.msno
# MAGIC     LEFT OUTER JOIN (  -- last known expiration date headed into this period
# MAGIC       SELECT
# MAGIC         x.subscription_id,
# MAGIC         y.membership_expire_date as last_at
# MAGIC       FROM (
# MAGIC         SELECT  -- last subscription transaction before start of this period
# MAGIC           subscription_id,
# MAGIC           MAX(transaction_date) as transaction_date
# MAGIC         FROM kkbox.transactions_enhanced
# MAGIC         WHERE transaction_date < '2017-03-01'
# MAGIC         GROUP BY subscription_id
# MAGIC         ) x
# MAGIC       INNER JOIN kkbox.transactions_enhanced y
# MAGIC         ON x.subscription_id=y.subscription_id AND x.transaction_date=y.transaction_date
# MAGIC       ) c
# MAGIC       ON b.subscription_id=c.subscription_id  
# MAGIC       ),
# MAGIC   activity (
# MAGIC     SELECT
# MAGIC       a.subscription_id,
# MAGIC       a.msno,
# MAGIC       b.date,
# MAGIC       CASE WHEN b.date > a.last_exp_at THEN 1 ELSE 0 END as after_exp,
# MAGIC       CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END as had_session,
# MAGIC       COALESCE(c.session_count, 0) as sessions_total,
# MAGIC       COALESCE(c.total_secs, 0) as seconds_total,
# MAGIC       COALESCE(c.num_uniq,0) as number_uniq,
# MAGIC       COALESCE(c.num_total,0) as number_total
# MAGIC     FROM activity_window a
# MAGIC     INNER JOIN kkbox.dates b
# MAGIC       ON b.date BETWEEN a.start_at AND a.end_at
# MAGIC     LEFT OUTER JOIN (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         date,
# MAGIC         COUNT(*) as session_count,
# MAGIC         SUM(total_secs) as total_secs,
# MAGIC         SUM(num_uniq) as num_uniq,
# MAGIC         SUM(num_25+num_50+num_75+num_985+num_100) as num_total
# MAGIC       FROM kkbox.user_logs
# MAGIC       GROUP BY msno, date
# MAGIC       ) c
# MAGIC       ON a.msno=c.msno AND b.date=c.date
# MAGIC     )
# MAGIC   
# MAGIC SELECT 
# MAGIC   subscription_id,
# MAGIC   msno,
# MAGIC   COUNT(*) as days_total,
# MAGIC   SUM(had_session) as days_with_session,
# MAGIC   COALESCE(SUM(had_session)/COUNT(*),0) as ratio_days_with_session_to_days,
# MAGIC   SUM(after_exp) as days_after_exp,
# MAGIC   SUM(had_session * after_exp) as days_after_exp_with_session,
# MAGIC   COALESCE(SUM(had_session * after_exp)/SUM(after_exp),0) as ratio_days_after_exp_with_session_to_days_after_exp,
# MAGIC   SUM(sessions_total) as sessions_total,
# MAGIC   COALESCE(SUM(sessions_total)/COUNT(*),0) as ratio_sessions_total_to_days_total,
# MAGIC   COALESCE(SUM(sessions_total)/SUM(had_session),0) as ratio_sessions_total_to_days_with_session,
# MAGIC   SUM(sessions_total * after_exp) as sessions_total_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(sessions_total * after_exp)/SUM(had_session * after_exp),0) as ratio_sessions_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(seconds_total) as seconds_total,
# MAGIC   COALESCE(SUM(seconds_total)/COUNT(*),0) as ratio_seconds_total_to_days_total,
# MAGIC   COALESCE(SUM(seconds_total)/SUM(had_session),0) as ratio_seconds_total_to_days_with_session,
# MAGIC   SUM(seconds_total * after_exp) as seconds_total_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(seconds_total * after_exp)/SUM(had_session * after_exp),0) as ratio_seconds_total_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_uniq) as number_uniq,
# MAGIC   COALESCE(SUM(number_uniq)/COUNT(*),0) as ratio_number_uniq_to_days_total,
# MAGIC   COALESCE(SUM(number_uniq)/SUM(had_session),0) as ratio_number_uniq_to_days_with_session,
# MAGIC   SUM(number_uniq * after_exp) as number_uniq_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_uniq * after_exp)/SUM(had_session * after_exp),0) as ratio_number_uniq_after_exp_to_days_after_exp_with_session,
# MAGIC   SUM(number_total) as number_total,
# MAGIC   COALESCE(SUM(number_total)/COUNT(*),0) as ratio_number_total_to_days_total,
# MAGIC   COALESCE(SUM(number_total)/SUM(had_session),0) as ratio_number_total_to_days_with_session,
# MAGIC   SUM(number_total * after_exp) as number_total_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(after_exp),0) as ratio_number_total_after_exp_to_days_after_exp,
# MAGIC   COALESCE(SUM(number_total * after_exp)/SUM(had_session * after_exp),0) as ratio_number_total_after_exp_to_days_after_exp_with_session
# MAGIC FROM activity
# MAGIC GROUP BY subscription_id, msno
# MAGIC ORDER BY msno;
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.test_act_features;

# COMMAND ----------

# MAGIC %md また、リスクのあるサブスクリプションのレコードを見逃していないことを確認しましょう。 これらのクエリは、それぞれゼロのカウントを返す必要があります。

# COMMAND ----------

# DBTITLE 1,Features for All Training Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM kkbox.train a
# MAGIC LEFT OUTER JOIN kkbox.train_act_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL

# COMMAND ----------

# DBTITLE 1,Features for All Testing Subscribers
# MAGIC %sql
# MAGIC 
# MAGIC SELECT COUNT(*)
# MAGIC FROM kkbox.test a
# MAGIC LEFT OUTER JOIN kkbox.test_act_features b
# MAGIC   ON a.msno=b.msno
# MAGIC WHERE b.msno IS NULL
