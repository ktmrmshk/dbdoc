# Databricks notebook source
# MAGIC %md 購読データを準備したので、KKBox音楽サービスで観察された顧客離脱のパターンを調べ始めます。 このノートでは、次のノートで行うより詳細な作業に備えて、顧客の離脱の一般的なパターンについて説明します。

# COMMAND ----------

# MAGIC %md **注意** このノートは2020年7月20日に改訂されました。

# COMMAND ----------

# MAGIC %md ##ステップ1: 環境を整える
# MAGIC 
# MAGIC このノートブックとその後のノートブックで使用するテクニックは、[生存分析](https://en.wikipedia.org/wiki/Survival_analysis#:~:text=Survival%20analysis%20is%20a%20branch,and%20failure%20in%20meical%20systems.)の領域から来ています。これらの技術をサポートするPythonのノートブックはいくつかありますが、ここでは、現在利用可能な生存分析ライブラリの中で最も人気のある[lifelines](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html)を利用します。 そのためには、まず、ライブラリをインストールして、クラスタにロードする必要があります。
# MAGIC 
# MAGIC **注意** 次のセルでは、MLランタイムを使用していないDatabricksクラスタでこのノートブックを実行していることを想定しています。 MLランタイムを使用している場合は、以下の[代替手順](https://docs.databricks.com/libraries.html#workspace-library)に従ってlifelinesライブラリをお使いの環境にロードしてください。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
dbutils.library.installPyPI('lifelines')
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,必要なライブラリの読み込み
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test

# COMMAND ----------

# MAGIC %md ##ステップ2: 母集団レベルの生存率の調査
# MAGIC 
# MAGIC サブスクリプションの全データセットを使って、メンバーが時間の経過とともにサブスクリプションサービスから脱落する様子を見てみましょう。 そのために、観測されたデータから簡単な統計計算を行って、ある時点（購読開始からの日数）での生存確率を特定する[Kaplan-Meier曲線](https://towardsdatascience.com/kaplan-meier-curves-c5768e349479#:~:text=The%20Kaplan%2DMeier%20estimator%20is,at%20%20a certaintime%20interval.)を導き出します。

# COMMAND ----------

# DBTITLE 1,サブスクリプションデータの取得
# retrieve subscription data to pandas DF
subscriptions_pd = spark.table('kkbox.subscriptions').toPandas()
subscriptions_pd.head()

# COMMAND ----------

# DBTITLE 1,人口に対する生存曲線の導出
kmf = KaplanMeierFitter(alpha=0.05) # calculate a 95% confidence interval
kmf.fit(subscriptions_pd['duration_days'], subscriptions_pd['churned'])

# COMMAND ----------

# MAGIC %md 前のステップの出力は、約310万件の購読レコードを使用してKMモデルをフィットさせたことを示していますが、そのうち約130万件は2017年4月1日の時点でまだアクティブでした。 (right-censored*という用語は、対象となるイベント、すなわち*i.e. * churn*が観測窓内で発生していないことを示しています)。 このモデルを使って、任意のサブスクリプションの生存時間の中央値を計算することができます。

# COMMAND ----------

# DBTITLE 1,生存時間の中央値の算出
median_ = kmf.median_survival_time_
median_

# COMMAND ----------

# MAGIC %md データセットに添付されている資料によると、KKBoxのメンバーは通常30日周期でサービスに加入しています。生存期間の中央値である184日は、ほとんどのお客様が最初に30日の期間で登録し、平均して6ヶ月間、毎月購読を更新した後に退会することを示しています。
# MAGIC 
# MAGIC モデルに、30日間の初期登録、1年間の契約、2年間の更新に対応するさまざまな値を渡すと、顧客がサービスを継続する、つまり*その時点を超えて生存する確率を計算することができます。

# COMMAND ----------

# DBTITLE 1,ある時点で生存している人口の割合
kmf.predict([30, 365, 730])

# COMMAND ----------

# MAGIC %md これをグラフ化すると、お客様の離脱率が契約年数に応じてどのように変化するかがわかります。

# COMMAND ----------

# DBTITLE 1,時間経過による生存率
# plot attributes
plt.figure(figsize=(12,8))
plt.title('All Subscriptions', fontsize='xx-large')

# y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':') # median line in red

# x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))
plt.axvline(30, color='gray', alpha=0.5, linestyle=':')  # 30-day gray dashed line
plt.axvline(180, color='gray', alpha=0.5, linestyle=':')  # 30-day gray dashed line
plt.axvline(365, color='gray', alpha=0.5, linestyle=':') # 1-year gray dashed line
plt.axvline(365*2, color='gray', alpha=0.5, linestyle=':') # 2-year gray dashed line

kmf.plot_survival_function()

# COMMAND ----------

# MAGIC %md ##ステップ3: 生存率の変化を調べる
# MAGIC 
# MAGIC 生存率の全体的なパターンは、KKBoxにおける顧客の減少について非常に説得力のあるストーリーを語っていますが、このパターンがサブスクリプションの属性によってどのように変化するかを調べることは興味深いことです。契約時の属性に注目することで、顧客獲得のための最大のインセンティブを提供している時に、アカウントの長期的な維持確率を示す変数を特定することができるかもしれません。そこで、サブスクリプションに関連する登録チャネルと、初期登録時に選択された初期支払方法および支払プランの日数を調査します。

# COMMAND ----------

# DBTITLE 1,初期属性を持つサブスクリプション
# MAGIC %sql  -- identify registration channel, initial payment method and initial payment plan days by subscription
# MAGIC DROP VIEW IF EXISTS subscription_attributes;
# MAGIC 
# MAGIC CREATE TEMP VIEW subscription_attributes 
# MAGIC AS
# MAGIC   WITH transaction_attributes AS ( -- get payment method and plan days for each subscriber's transaction date
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         FIRST(b.payment_method_id) as payment_method_id,
# MAGIC         FIRST(b.payment_plan_days) as payment_plan_days
# MAGIC       FROM (  -- base transaction dates
# MAGIC         SELECT 
# MAGIC           msno,
# MAGIC           transaction_date as trans_at,
# MAGIC           MAX(membership_expire_date) as expires_at
# MAGIC         FROM kkbox.transactions
# MAGIC         GROUP BY msno, transaction_date
# MAGIC         ) a
# MAGIC       INNER JOIN kkbox.transactions b
# MAGIC         ON  a.msno=b.msno AND
# MAGIC             a.trans_at=b.transaction_date AND 
# MAGIC             a.expires_at=b.membership_expire_date
# MAGIC       WHERE b.payment_plan_days > 0
# MAGIC       GROUP BY
# MAGIC         a.msno,
# MAGIC         a.trans_at
# MAGIC       )
# MAGIC   SELECT
# MAGIC     m.*,
# MAGIC     n.payment_method_id as init_payment_method_id,
# MAGIC     n.payment_plan_days as init_payment_plan_days,
# MAGIC     COALESCE(CAST(o.registered_via as String), 'Unknown') as registered_via
# MAGIC   FROM kkbox.subscriptions m
# MAGIC   INNER JOIN transaction_attributes n
# MAGIC     ON  m.msno=n.msno AND      -- match on customer
# MAGIC         m.starts_at=n.trans_at -- and transaction date at start of transaction
# MAGIC   LEFT OUTER JOIN kkbox.members o  -- membership info (assume stable across subscription)
# MAGIC     ON m.msno=o.msno
# MAGIC   ORDER BY m.subscription_id

# COMMAND ----------

# DBTITLE 1,初期属性を持つサブスクリプション
# capture output to Spark DataFrame
subscriptions = spark.table('subscription_attributes')

# capture output to pandas DataFrame
subscriptions_pd = subscriptions.toPandas()
subscriptions_pd.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC 属性がサブスクリプションに関連付けられたので、お客様がサービスを利用する際の登録経路を調べてみましょう。
# MAGIC 初期属性を持つサブスクリプション

# COMMAND ----------

# DBTITLE 1,登録チャネル別会員数
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   registered_via,
# MAGIC   COUNT(DISTINCT msno) as members
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY registered_via
# MAGIC ORDER BY members DESC

# COMMAND ----------

# MAGIC %md 番号付きのチャンネルについては何もわかりませんが、チャンネル7と9が圧倒的に人気があることは明らかです。 他のいくつかのチャンネルも人気がありますが、わずかな数の加入者しかいないチャンネルもいくつかあります。
# MAGIC 
# MAGIC 分析をシンプルにするために、チャンネル13、10、16を除外してみましょう。これらのチャンネルは、合計してもユニークユーザー数の0.3%未満にしかなりません。 このようにして、残りのチャンネルごとに独立した曲線を表示して、生存曲線を再検討することができます。

# COMMAND ----------

# DBTITLE 1,登録チャネル別生存率
# eliminate nominal channels
channels_pd = subscriptions_pd[~subscriptions_pd['registered_via'].isin(['10','13','16'])]

# configure the plot
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Registration Channel', fontsize='xx-large')

# configure the x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# configure the y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

# graph each curve on the plot
for name, grouped_pd in channels_pd.groupby('registered_via'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Channel {0}'.format(name)
      )
    kmf.plot(ax=ax)

# COMMAND ----------

# MAGIC %md これらの異なる曲線を解釈しようとする前に、それらが統計的に異なるかどうかを評価することが重要です。 各曲線を他の曲線と比較すると、[log-rank test](https://en.wikipedia.org/wiki/Logrank_test)を使用して、これらの曲線が互いに異ならない確率を計算することができます。
# MAGIC 
# MAGIC **注意** 以下の呼び出しにt_0の引数を追加することで、ここで示されているようにすべての時間に渡ってではなく、特定の時点で各曲線に対して同じメトリクスを計算することができます。

# COMMAND ----------

# DBTITLE 1,確率曲線が同じになる（全体）
log_rank = pairwise_logrank_test(channels_pd['duration_days'], channels_pd['registered_via'], channels_pd['churned'])
log_rank.summary

# COMMAND ----------

# DBTITLE 1,確率曲線は同じ(184日目)
log_rank = pairwise_logrank_test(channels_pd['duration_days'], channels_pd['registered_via'], channels_pd['churned'], t_0=184)
log_rank.summary

# COMMAND ----------

# MAGIC %md 全体的に見ても、特に上述の生存期間の中央値である184日目において、これらの曲線のほとんどが互いに有意に異なっています（ほぼすべてのp値が<0.05であることからもわかります）。このことから、上のグラフの異なる表現には意味があることがわかります。 しかし、どのようにして？ ナンバリングされたチャネルに関する追加情報がなければ、ある顧客が他の顧客よりも高い離脱率を示している理由について、説得力のあるストーリーを語ることは困難です。 しかし、KKBoxは、顧客獲得活動の効果を最大化するために、なぜあるチャネルがより高い定着率を持つように見えるのかを探り、各チャネルに関連するコストの違いを調べることをお勧めします。
# MAGIC 
# MAGIC では、これと同じ分析を、サブスクリプションが作成されたときに使用された支払い方法について行ってみましょう。

# COMMAND ----------

# DBTITLE 1,初回支払方法別会員数
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   init_payment_method_id,
# MAGIC   COUNT(DISTINCT msno) as customers
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY init_payment_method_id
# MAGIC ORDER BY customers DESC

# COMMAND ----------

# MAGIC %md データセットに含まれる支払方法の数は非常に多い。 人気のあるチャンネルとそうでないチャンネルが明確に分かれていた登録チャンネルの場合とは異なり、支払い方法による購読数の減少はより緩やかです。 この点を考慮して、ある支払い方法に関連する会員数が10,000人という任意のカットオフ値を設定し、分析の対象とします。

# COMMAND ----------

# DBTITLE 1,一般的な初期支払方法に関連するサブスクリプションデータを取得する
payment_methods_pd = spark.sql('''
  SELECT
    duration_days,
    churned,
    init_payment_method_id
  FROM subscription_attributes
  WHERE init_payment_method_id IN (
    SELECT
      init_payment_method_id
    FROM subscription_attributes
    GROUP BY init_payment_method_id
    HAVING COUNT(DISTINCT msno)>10000
    )''').toPandas()

# COMMAND ----------

# DBTITLE 1,初期支払方法別生存率
# configure the plot
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Initial Payment Method', fontsize='xx-large')   

# configure the y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

# configure the x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# calculate the surival rates 
for name, grouped_pd in payment_methods_pd.groupby('init_payment_method_id'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Method {0}'.format(name)
      )
    _ = kmf.plot(ax=ax)
    _.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md 人気のある決済方法だけを見ても、上のグラフは先ほどのグラフよりもかなり忙しくなっています。これは、2つの支払い方法の間の統計的な違いを慎重に検討してから、あまり難しい結論を出すべきチャートだと思います。 しかし、紙面の都合上、今はこのチャートを解釈し、下のセルにあるペアワイズの統計比較を取り上げます。
# MAGIC 
# MAGIC 理由を推測する知識はありませんが、いくつかの方法でドロップオフ率が大きく異なるのは非常に興味深いことです。 例えば、グラフの一番上にある34番の方法と一番下にある35番の方法を比較してみましょう。 このような支払い方法の違いは、長期的な支払い能力の違いを示しているのでしょうか。例えば、クレジットカードは高所得者やクレジットスコアの高い人に、カードは低所得者やクレジットスコアの低い人に、というように。 あるいは、これらの初期支払方法のいくつかは、最初の30日間、お客様が何らかの方法で支払いを放棄したり、割引価格でサービスを受けたりするクーポン券に結びつけることもできます。 その後、通常価格でのお支払いをお願いした場合、お客様は最初からサービスに大きな投資をしていなかったため、サービスをやめてしまう可能性があります。 (ここで重要なのは、私たちが知っているのは最初に使われた支払い方法だけで、その後に使われた支払い方法は知らないということです。） 繰り返しになりますが、ビジネスに関する知識がなければ、推測することしかできません。しかし、ここで示された大きな違いを考えると、これは顧客獲得の一側面として、より詳細に調査する価値があるでしょう。
# MAGIC 
# MAGIC ここでは、統計的に異なると考えられるほどの差がない曲線に限定して、ペアワイズ比較を行っています。

# COMMAND ----------

# DBTITLE 1,確率曲線は同じ
log_rank = pairwise_logrank_test(payment_methods_pd['duration_days'], payment_methods_pd['init_payment_method_id'], payment_methods_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md Log Rank 検定の結果から、支払方法 22 と 32 は統計的に差がないことがわかります（20 と 40 も同様）。
# MAGIC 
# MAGIC 次に、サブスクリプションの開始時に支払いプランに設定された日数を調べてみましょう。 これは、連続的にも離散的にも見ることができる、奇妙なサブスクリプションの属性です。ここでは離散的に扱い、後の分析でどのように扱うかを考えてみましょう。
# MAGIC 
# MAGIC この時点で、この分析のパターンはおなじみだと思います。

# COMMAND ----------

# DBTITLE 1,初回支払いプラン日数別のメンバー
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   init_payment_plan_days,
# MAGIC   COUNT(DISTINCT msno) as customers
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY init_payment_plan_days
# MAGIC ORDER BY customers DESC

# COMMAND ----------

# DBTITLE 1,プロット 人気 初期費用 プレイ日数
payment_plan_days_pd = spark.sql('''
  SELECT
    duration_days,
    churned,
    init_payment_plan_days
  FROM subscription_attributes
  WHERE init_payment_plan_days IN (
    SELECT
      init_payment_plan_days
    FROM subscription_attributes
    GROUP BY init_payment_plan_days
    HAVING COUNT(DISTINCT msno)>10000
    )''').toPandas()
   
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Initial Payment Plan Days', fontsize='xx-large')

plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# calculate the surival rates 
for name, grouped_pd in payment_plan_days_pd.groupby('init_payment_plan_days'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Days {0}'.format(name)
      )
    _ = kmf.plot(ax=ax)
    _.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md ほとんどのお客様が1回目の更新時に大きく脱落しているように見えます。KKBoxが従来のプランとしている30日プランに比べて、7日プランと10日プランでは初回更新時に大幅な落ち込みが見られるなど、すべてのプランで同じ落ち込み率ではない。また、興味深いのは、100日プランでは最初の更新時に顧客数が急減している（生存率は30日プランの顧客数を下回る）のに対し、90日プランの顧客は契約ライフサイクルの後半まで全く異なる軌跡をたどっていることだ。
# MAGIC 
# MAGIC これらの曲線の信頼区間は、初期のものよりもはっきりしていますが、統計的には、すべての曲線が有意です。

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same
log_rank = pairwise_logrank_test(payment_plan_days_pd['duration_days'], payment_plan_days_pd['init_payment_plan_days'], payment_plan_days_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md 最後に、過去に解約したことのある加入者が、1回しか加入したことのない加入者とは異なる経路をたどる可能性について考えてみましょう。

# COMMAND ----------

# DBTITLE 1,プロットの先行予約数
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Prior Subscription Count', fontsize='xx-large')

plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))
 
for name, grouped_pd in subscriptions_pd.groupby('prior_subscriptions'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Prior Subscriptions {0}'.format(name)
      )
    kmf.plot(ax=ax)

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same
log_rank = pairwise_logrank_test(subscriptions_pd['duration_days'], subscriptions_pd['prior_subscriptions'], subscriptions_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md 過去の契約数が増加するにつれて、各カテゴリーに該当する加入者の数は減少し、信頼区間はますます大きくなる。過去の契約数が少ない加入者は高い割合で維持されるという一般的なパターンがあるように見えるが、それもある時点までの話である。 しかし、それはある時点までのことであり、その後は予備知識の数がドロップアウトからの保護にはならないようです。 統計的に有意ではないので、この物語をあまり重視しない方が良いと思いますが、それでもKKBoxでは興味深い調査ができるかもしれません。
