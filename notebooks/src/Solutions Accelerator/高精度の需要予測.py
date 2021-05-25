# Databricks notebook source
# MAGIC %md # Facebook ProphetとSparkを活用した高精度需要予測
# MAGIC 
# MAGIC 時系列データ予測技術の進化は、小売業界においてより信頼性のある需要予測を可能としました。今や、適切な精度およびスピードで予測値を提供し、ビジネスサイドで商品在庫を正確に調整できる様にすることが、新たな課題になっています。これらの課題に直面している多くの企業が、[Apache Spark™](https://databricks.com/spark/about)と[Facebook Prophet](https://facebook.github.io/prophet/)を活用することで、これまでのソリューションにあった精度およびスケーラビリティの課題を克服しています。
# MAGIC 
# MAGIC 本ノートブックでは、時系列データ予測の重要性を議論し、サンプル時系列データの可視化を行います。そして、簡単なモデルを構築してFacebook Prophetの使用法を説明します。単一のモデル構築に慣れた後で、ProphetとApache Spark™を結合させ、どの様にして数百のモデルを一度に学習するのかをお見せします。これにより、これまでは実現困難であった、個々の製品、店舗レベルでの正確な予測が可能となります。
# MAGIC 
# MAGIC <table>
# MAGIC   <tr><th>作者</th><th>ナレッジコミュニケーション / Databricks Japan</th></tr>
# MAGIC   <tr><td>日付</td><td>2021/03/11</td></tr>
# MAGIC   <tr><td>バージョン</td><td>1.1</td></tr>
# MAGIC   <tr><td>クラスター</td><td>8.0ML</td></tr>
# MAGIC   <tr><td>ライブラリ</td><td>FBProphet 0.7.1<br>pystan 2.19.1.1</td></tr>
# MAGIC </table>
# MAGIC 
# MAGIC <img style="margin-top:25px;" src="https://sajpstorage.blob.core.windows.net/workshop20210205/databricks-logo-small-new.png" width="140">

# COMMAND ----------

# MAGIC %md ## 正確かつタイムリーな需要予測はこれまで以上に重要となっています
# MAGIC 
# MAGIC 小売業界で成功するためには、製品・サービスの需要を予測するために時系列データを分析するスピードと精度を改善することが重要です。もし、多くの商品を店舗に配置してしまうと、棚や倉庫のスペースは圧迫され、商品は期限切れとなり、彼らの経営資源は在庫に縛られてしまい、製造業、あるいは顧客の行動パターンの変化によってもたらされる新たな機会に対応することが不可能となります。また、商品を少なく配置してしまうと、顧客は必要な商品が買えないということになります。予測のエラーは小売業にとって収益ロスになるだけでなく、長期間に渡って顧客のフラストレーションを増加させ、競合に走ってしまうことになりかねません。

# COMMAND ----------

# MAGIC %md ## より正確な時系列データ予測の手法とモデルへの期待が高まっています
# MAGIC 
# MAGIC かつて、ERPシステムとサードパーティのソリューションはシンプルな時系列モデルに基づく需要予測機能を提供していました。しかし、技術の進歩と業界全体におけるプレッシャーから、多くの小売業はこれまで使っていた線形モデルや従来のアルゴリズムから、その先に目を向け始めています。
# MAGIC 
# MAGIC データサイエンスコミュニティでは、[Facebook Prophet](https://facebook.github.io/prophet/)によって提供される様な新たなライブラリが人気を得ており、多くの企業はこれらの機械学習モデルを時系列予測に適用できないか模索しています。
# MAGIC 
# MAGIC <img width="200" src="https://databricks.com/wp-content/uploads/2020/01/FB-Prophet-logo.png">

# COMMAND ----------

# MAGIC %md ## 時系列データにおける需要の季節性の可視化
# MAGIC 
# MAGIC 個々の店舗、製品に対する高精度の需要予測を行うためにどの様にProphetを使うのかをデモンストレーションするために、Kaggleで公開されている[データセット](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)を使います。これは、10店舗における、50アイテムに対する5年間の日別の販売データです。

# COMMAND ----------

# MAGIC %md ## 以降のノートブックの流れ
# MAGIC <br>
# MAGIC 1. データのインポートとテーブル作成
# MAGIC 2. データの概要把握のための探索的データ分析(EDA: Exploratory Data Analysis)
# MAGIC 3. モデル #1 - Prophet を用いて販売数予測 (全体トレンドから予測)
# MAGIC 4. モデル #2 - Prophet を用いて販売数予測 (店舗・アイテムの組み合わせを考慮) : スターバックスのユースケースで実装されている大規模データの処理に最適化されたモデル

# COMMAND ----------

# MAGIC %md ## 1. データのインポートとテーブル作成

# COMMAND ----------

# MAGIC %md ### 前準備(ライブラリのインストール)

# COMMAND ----------

# MAGIC %pip install pystan==2.19.1.1

# COMMAND ----------

# MAGIC %pip install FBProphet

# COMMAND ----------

# MAGIC %md
# MAGIC ### 前準備(データのダウンロード)
# MAGIC 1. [Kaggle](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)から ```train.csv``` をダウンロードする
# MAGIC 2. ```/FileStore/tables/demand_forecast/train/``` に ```train.csv``` をアップロードする。
# MAGIC 
# MAGIC ** アップロード手順 **
# MAGIC 1. 上部メニューのFile > Upload Dataを選択<br>
# MAGIC ![](https://sajpstorage.blob.core.windows.net/notebook20210311-demand-forecast/Screen Shot 2021-03-11 at 21.05.06.png)
# MAGIC 
# MAGIC 2. パスに「`/FileStore/tables/demand_forecast/train/`」を指定し、ダウンロードしたCSVをドラッグアンドドロップする
# MAGIC ![](https://sajpstorage.blob.core.windows.net/notebook20210311-demand-forecast/Screen Shot 2021-03-11 at 21.06.04.png)

# COMMAND ----------

# # ライブラリのインポート(Databricks Runtimeの際に実行)
# dbutils.library.installPyPI('FBProphet', version='0.5') # 最新バージョンはこちら: https://pypi.org/project/fbprophet/
# dbutils.library.installPyPI('holidays','0.9.12') # fbprophet 0.5 のissueへの対応 https://github.com/facebook/prophet/issues/1293
# dbutils.library.restartPython()

# # Prophet を実行する際に出るメッセージを非表示にするオプション
# import logging
# logger = spark._jvm.org.apache.log4j
# logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

import re

# Username を取得。
username_raw = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
# Username の英数字以外を除去し、全て小文字化。Usernameをファイルパスやデータベース名の一部で使用可能にするため。
username = re.sub('[^A-Za-z0-9]+', '', username_raw).lower()

# データベース名を生成。
db_name = f"prophet_spark_demo_{username}"

# データベースの準備
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
spark.sql(f"USE {db_name}")

print("database name: " + db_name)

# COMMAND ----------

# 本ノートブックのSQLがSpark3+で動作するために必要な設定
spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY")

# COMMAND ----------

# MAGIC %md ### データの読み込み

# COMMAND ----------

from pyspark.sql.types import *

# 今回利用するデータのスキーマを指定 (型修正オプションで自動読み取りも可能)
schema = StructType([
  StructField("date", DateType(), False),
  StructField("store", IntegerType(), False),
  StructField("item", IntegerType(), False),
  StructField("sales", IntegerType(), False)
])

# FileStore に格納されているCSVを読み込み
inputDF = spark.read.format("csv")\
.options(header="true", inferSchema="true")\
.load("/FileStore/tables/demand_forecast/train/train.csv", schema=schema)

# クエリを発行可能な状態にするために、一時ビューを作成
inputDF.createOrReplaceTempView('history')
history = inputDF

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. データ概要把握のためのEDA
# MAGIC 
# MAGIC 読み込んだデータは以下の構成となっています
# MAGIC 
# MAGIC * `date` 日付
# MAGIC * `store` 店舗番号
# MAGIC * `item` 商品番号
# MAGIC * `sales` 販売点数

# COMMAND ----------

# 件数の確認
history.count()

# COMMAND ----------

# 内容の表示
display(history)

# COMMAND ----------

# MAGIC %md
# MAGIC データブリックスのワークスペースにおいては、`%sql`のマジックコマンドを書くことで、セルレベルで言語をSQLに切り替えることができます。これにより、作成した一時ビューに対してSQL文を直接発行して問い合わせを行うことが可能です。

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM history

# COMMAND ----------

# MAGIC %md
# MAGIC Python のコードの中に埋め込むことで、テーブルに対して同様の処理を行うことも可能です。

# COMMAND ----------

query = """
SELECT * FROM history
"""
display(spark.sql(query))

# COMMAND ----------

# MAGIC %md
# MAGIC また、以下のような機能も提供しています。
# MAGIC 
# MAGIC - 左上の Revision History より変更履歴を確認でき、任意の時点のノートブックに戻すことも容易です
# MAGIC - 左上のCommentsから、コードのキーワードレベルでコメントを残すことが可能です
# MAGIC - 同じノートブックを同時に複数人で編集することができます

# COMMAND ----------

# MAGIC %md 以下では様々な切り口でデータを確認していきます

# COMMAND ----------

# MAGIC %sql -- 日付:1826日 > 約5年分のデータ
# MAGIC SELECT COUNT(DISTINCT date) FROM history

# COMMAND ----------

# MAGIC %sql -- store の要素数 > 10 店舗
# MAGIC SELECT COUNT(DISTINCT store) FROM history

# COMMAND ----------

# MAGIC %sql -- item の要素数 > 50 アイテム
# MAGIC SELECT COUNT(DISTINCT item) FROM history

# COMMAND ----------

# MAGIC %sql -- 時系列データの範囲 > 2013-2017年の5年分
# MAGIC SELECT DISTINCT YEAR(date) AS YEAR FROM history ORDER BY YEAR

# COMMAND ----------

# MAGIC %md Databricksの[コラボレーティブノートブック](https://databricks.com/product/collaborative-notebooks)にビルトインされている可視化機能を使うことで、チャート上にマウスカーソルを持っていくだけでデータの値を確認することができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 年毎の集計。店舗全体では年々増加傾向にあることがわかります。
# MAGIC -- 与えられたデータセットにおいて、週単位や月単位の変動も織り込んだ予測を行うことができれば、精度の高めることが可能となります。
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'YYYY') as year,
# MAGIC   SUM(sales) as sales
# MAGIC FROM history
# MAGIC GROUP BY TRUNC(date, 'YYYY')
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 月ごとのトレンド。月ごとでは単純に増加傾向にはないことがわかります。代わりに、夏を山、冬を谷とした明らかな季節性があることがわかります。
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM history
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 年毎・曜日ごとのトレンド
# MAGIC -- どの年でも日曜日(weekday 0)にピークとなり、月曜(weekday 1)に売り上げが落ち、徐々に日曜に向けて売上増加していくことがわかります。
# MAGIC -- このトレンドは5年間の観察期間全体でかなり安定しているように見受けられます。
# MAGIC 
# MAGIC SELECT -- 年毎で、曜日別売上の平均を算出
# MAGIC   YEAR(date) as year,
# MAGIC   CAST(DATE_FORMAT(date, 'u') as Integer) % 7 as weekday, -- 日曜から土曜まで0-6となるように曜日を導出
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT -- 日ごとの売り上げで集計したテーブルをサブクエリに指定します
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, CAST(DATE_FORMAT(date, 'u') as Integer) -- 年毎・曜日ごとで集計
# MAGIC ORDER BY year, weekday;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 店舗毎・月毎のトレンド
# MAGIC 
# MAGIC SELECT -- 店舗毎で、月売上の平均を算出
# MAGIC   MONTH(month) as month,
# MAGIC   store,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT -- 月ごとの売り上げで集計したテーブルをサブクエリに
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     store,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   GROUP BY TRUNC(date, 'MM'), store
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), store
# MAGIC ORDER BY month, store;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- アイテム毎・月毎のトレンド
# MAGIC 
# MAGIC SELECT -- アイテム毎で、月売上の平均を算出
# MAGIC   MONTH(month) as month,
# MAGIC   item,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT -- 月ごとの売り上げで集計したテーブルをサブクエリに
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     item,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM history
# MAGIC   GROUP BY TRUNC(date, 'MM'), item
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), item
# MAGIC ORDER BY month, item

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. モデル #1 - Prophet を用いて販売数予測　(全体トレンドから予測)
# MAGIC 
# MAGIC 上のチャートに現れている通り、このデータは、年レベル、週レベルでの季節性を伴って上昇傾向を示していることがわかります。Prophetは、この様にデータにおいて複数の重なり合うするパターンを取り扱うために設計されました。
# MAGIC 
# MAGIC [Prophet Python API Quick Start](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
# MAGIC 
# MAGIC - 統計の知識がなくても予測を作成できる
# MAGIC - 年・週・日 単位で強い周期性を持つ時系列データの予測に強い : 時系列データ = トレンド + 周期性 (年・月・週など)
# MAGIC - ドメイン知識をもとに、利用者が予測モデルをチューニング可能
# MAGIC - 傾向を見るうえでネックとなるデータの抜けに強い
# MAGIC - 外れ値(極端に売り上げが高い/低い)をうまく扱える
# MAGIC 
# MAGIC Facebook Prophetはscikit-learn APIのパターンに従っているので、sklearnの経験がある人は簡単に利用できます。まず、2つの列を持ったpandasデータフレームをインプットとして渡す必要があります:<br><br>
# MAGIC 
# MAGIC - 第1の列は日付
# MAGIC - 第2の列は予測すべき値(本ケースにおいては売り上げ)
# MAGIC 
# MAGIC データが適切な形でフォーマットされているのであれば、モデル構築は容易です。
# MAGIC 
# MAGIC まずは、店舗やアイテム個別の組み合わせを考慮しない全体的トレンドに基づくモデルを構築します。以降では、Prophetを用い、予測誤差の範囲の広さ(信頼区間)を95%とし、モデルのパラメータを定義していきます。
# MAGIC 
# MAGIC [19\-3\. 95％信頼区間のもつ意味 \| 統計学の時間 \| 統計WEB](https://bellcurve.jp/statistics/course/8891.html)

# COMMAND ----------

# モデルの定義
def define_prophet_model(params):
  model = Prophet(
      interval_width=params["interval_width"],
      growth=params["growth"],
      daily_seasonality=params["daily_seasonality"],
      weekly_seasonality=params["weekly_seasonality"],
      yearly_seasonality=params["yearly_seasonality"],
      seasonality_mode=params["seasonality_mode"]
      )
  return model

# 予測
def make_predictions(model, number_of_days):
  return model.make_future_dataframe(periods=number_of_days, freq='d', include_history=True)

# COMMAND ----------

from pyspark.sql.functions import to_date, col
from pyspark.sql.types import IntegerType

# 学習用の pandas df を用意
# 2015/1/1以降のデータから1%のサンプリングを実施
history_sample = (
  history
  .where(col("date") >= "2015-01-01")
  .sample(fraction=0.01, seed=123)
)

history_pd = (
  history_sample
  .toPandas()
  .rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]
)


# fbprophetを利用した場合のメッセージを非表示に
import logging
logging.getLogger('py4j').setLevel(logging.ERROR)

# モデルの定義
from fbprophet import Prophet

params = {
      "interval_width": 0.95,
      "growth": "linear", # linear or logistic
      "daily_seasonality": False,
      "weekly_seasonality": True,
      "yearly_seasonality": True,
      "seasonality_mode": "multiplicative"
    }

model = define_prophet_model(params)

# COMMAND ----------

# MAGIC %md 全体傾向を表現するため、日付ごとに売り上げを集計したデータを準備します

# COMMAND ----------

import pyspark.sql.functions as psf

aggregated_history_sample = (
  history_sample
  .select("date", "sales")
  .groupBy("date")
  .agg( psf.sum("sales").alias("sales") )
)

aggregated_history_pd = (
  aggregated_history_sample
  .toPandas()
  .rename(columns={'date':'ds', 'sales':'y'})[['ds','y']]
)

# COMMAND ----------

aggregated_history_pd

# COMMAND ----------

# MAGIC %md
# MAGIC モデル及びデータ準備できたので、向こう90日間の予測を行います。ds が日付、yhat が予測値、その他の項目はトレンドや周期性などを表しています。

# COMMAND ----------

# 過去のデータをモデルに学習させる
model.fit(aggregated_history_pd)

# 過去のデータと先90日間を含むデータフレームを定義
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# データセット全体に対して予測実行
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# それぞれのdataframeのサイズを確認する
print("history_pd:{} aggregated_history_sample:{} future_pd:{} forecast_pd:{}".format(len(history_pd), len(aggregated_history_pd), len(future_pd), len(forecast_pd)))

# COMMAND ----------

# MAGIC %md
# MAGIC 結果をグラフで出力します。
# MAGIC - 上は販売推移の大まかなトレンドです。販売数は上昇傾向ですが、2016年10月ころから鈍化しています。
# MAGIC - 真ん中は週単位の周期性です。月曜に販売が落ち、土日に向けて上昇していく傾向が見られます。
# MAGIC - 下は年単位の周期性です。年末から2月ころが少なく、夏に向けて上昇、9―11月にかけて落ち着く、という傾向が見られます

# COMMAND ----------

trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 実際のデータと予測されたデータがどのように並んでいるかをグラフ化します。<br>
# MAGIC 見やすくするために、過去1年の履歴データと予測結果のみに絞ります。

# COMMAND ----------

predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')

# 出力されるデータを過去1年と予測期間のみに絞る
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

# MAGIC %md
# MAGIC 黒い点は実績を表し、濃い青色の線は予測値、明るい青の帯は95％の信頼性区間を表します。<br>
# MAGIC このグラフでは作成したモデルが良いモデルかどうかの判断を付けづらいので、以下を予測モデルの評価指標として用います<br>
# MAGIC <br>
# MAGIC 
# MAGIC **平均絶対誤差 (MAE: Mean Absolute Error)**<br>
# MAGIC 各データにおいて実測値と予測値の誤差の絶対値を取り、平均をとったもの。<br>
# MAGIC RMSE に比べて外れ値の影響を受けにくいと言われる。
# MAGIC 
# MAGIC **平均二乗誤差 (MSE: Mean Squared Error)**<br>
# MAGIC 誤差を二乗し、平均をとったもの
# MAGIC 
# MAGIC **二乗平均平方根誤差 (RMSE: Root Mean Squared Error)**<br>
# MAGIC MSE の平方根をとったもの。MAEに比べて大きな誤差を厳しく評価する特徴がある。

# COMMAND ----------

# DatetimeをDateに変換します
forecast_pd['ds'] = forecast_pd['ds'].dt.date

# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# 比較のために過去の実績と予測を取得
actuals_pd = aggregated_history_pd[ aggregated_history_pd['ds'] < date(2018, 1, 1) ]['y']
predicted_pd = forecast_pd[ forecast_pd['ds'] < date(2018, 1, 1) ]['yhat']

# 精度指標の計算
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

print('-----------------------------')
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md
# MAGIC FBProphet を用いる際には、ここから更にチューニングし、モデルの精度を上げていくことがほとんどですが、今回はチューニングをスキップして、処理をスケーリングできるかどうかという課題に焦点を当てて、モデルを構築します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. モデル #2 - Prophet を用いて販売数予測 (店舗・アイテムの組み合わせ考慮)
# MAGIC 
# MAGIC 上では、単一の時系列予測モデルを構築できることを示しましたが、次にApache Sparkの力を借りて、我々の能力を更に増強します。我々の目的は、データセット全体に対する単一の予測を行うことではなく、個々の製品・店舗の組み合わせに対して、数百のモデルを構築し予測を行うことです。通常、これを逐次的に実行した場合には、信じられないほど時間を要することになります。
# MAGIC 
# MAGIC データサイエンティストは、[Apache Spark](https://databricks.com/spark/about)の様な分散データ処理エンジンを使い、大量モデルをトレーニングするという課題に頻繁に取り組んでいます。Sparkクラスターを活用することで、クラスターにおける個々のワーカーノードは並列でモデルのサブセットを学習することができます。これにより、一連の時系列モデルの学習に要する時間を大幅に削減することができます。
# MAGIC 
# MAGIC もちろん、ワーカーノード(コンピューター)のクラスターにおけるモデル学習は、より多くのクラウドインフラストラクチャー及びそのコストを必要とします。しかし、オンデマンドでのクラウドリソースの高い可用性により、企業は必要に応じて迅速かつ柔軟にリソースを確保できます。これによって、自前で長い期間資産を持つことなしに、高いスケーラビリティを確保できます。
# MAGIC 
# MAGIC Sparkにおける分散データ処理の鍵となるメカニズムが[データフレーム](https://databricks.com/glossary/what-are-dataframes)です。データをSparkデータフレームに読み込むことで、データはクラスターにおけるノードに分散されます。これにより各ワーカーは並列でデータのサブセットを処理できる様になり、処理全体に要する時間を削減することできます。
# MAGIC 
# MAGIC データにおけるキーの値(本ケースでは店舗とアイテムの組み合わせ)に基づきグルーピングを行い、特定のワーカーノードにこれらのキー値に対応する時系列データを分配します。

# COMMAND ----------

# MAGIC %md
# MAGIC 今回利用しているデータセットは数10MBですので、処理速度がネックになることは考えにくいですが、実ビジネスでは、例えば天候データや各種イベントなども考慮に入れたうえで予測を行いたい、といったケースが出てきます。
# MAGIC 
# MAGIC 以下では、ペタバイト級のデータセットにも対応可能にするために、スターバックスがとったアプローチでモデルを作成します。

# COMMAND ----------

sql_statement = '''
  SELECT
    store,
    item,
    date as ds,
    sales as y
  FROM
    history
  '''

store_item_history = (
  spark    
  .sql( sql_statement )
  .repartition(sc.defaultParallelism, ['store', 'item'])
).cache()

# COMMAND ----------

display(store_item_history)

# COMMAND ----------

# MAGIC %md
# MAGIC 店舗/アイテム/日付レベルで集計されたデータを使用して、FBProphetにデータを渡す必要があります。そのデータをサブセットとしてモデルをトレーニング・予測する流れとなります。
# MAGIC 
# MAGIC 予測結果は次のような構造のデータセットとして返されるものとします。
# MAGIC 
# MAGIC - `ds` 日付
# MAGIC - `store` 店舗ID
# MAGIC - `item` アイテムID
# MAGIC - `y` 実績値
# MAGIC - `yhat` 予測値
# MAGIC - `yhat_upper` 予測値上限
# MAGIC - `yhat_lower` 予測値下限

# COMMAND ----------

from pyspark.sql.types import *

result_schema =StructType([
  StructField('ds',DateType()),
  StructField('store',IntegerType()),
  StructField('item',IntegerType()),
  StructField('y',FloatType()),
  StructField('yhat',FloatType()),
  StructField('yhat_upper',FloatType()),
  StructField('yhat_lower',FloatType())
  ])

# COMMAND ----------

# MAGIC %md
# MAGIC 店舗とアイテムに基づいて、適切に時系列データをグルーピングすることで、個々のグループに対してモデルを構築することができます。これを行うためには、pandasにおけるユーザー定義関数(UDF)を利用できます。これにより、データフレームにおける各グループに対して、カスタム関数を適用することが可能となります。
# MAGIC 
# MAGIC このUDFは、個々のグループに対してモデルをトレーニングするだけではなく、モデルから得られる予測値も取得します。この関数は個々のグループとは独立して、データフレームにおける各グループを学習・予測しますが、最終的な結果は一つのデータフレームにまとめる形となります。これにより、店舗・アイテムレベルでの予測を生成しつつも、結果を単一のデータセットとして分析者に提示することが可能になります。

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
def forecast_store_item( history_pd ):
  
  # #1と同じロジックでモデルを作成
  # --------------------------------------
  # 欠損値を落とす(サブセットのデータ数によっては欠損値補完する必要あり)
  history_pd = history_pd.dropna()
  
  # モデル作成
  model = Prophet(
    interval_width=0.95,
    growth='linear', # linear or logistic
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # モデル学習
  model.fit( history_pd )
  # --------------------------------------
  
  # #1と同じロジックで予測
  # --------------------------------------
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # サブセットを結合
  # --------------------------------------
  # 予測から関連フィールドを取得
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # 履歴から関連するフィールドを取得
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # 履歴と予測を結合
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # データセットから店舗と品番を取得
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # データセットを返す
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  



# COMMAND ----------

# MAGIC %md
# MAGIC モデリング・予測を各データセット(店舗・アイテムごとの各データセットで行っているので、<br>
# MAGIC 最終的に予測結果を結合する処理を挟んでいますが、ベースとなる処理は#1のモデルとほとんど同じです。<br>
# MAGIC 処理は標準的なpandasデータフレーム操作で構成されています。

# COMMAND ----------

from pyspark.sql.functions import current_date

results = (
  store_item_history    
  .groupBy('store', 'item') # 分割して予測したいカラムを定義
  .apply(forecast_store_item) # 先のセルで定義した関数を適用 (モデリングと予測)  <===== store, itemでGroupByしたDataframeに対して、上記のUDFを適用し、store, itemごとにprophetモデルを作成、推定している。
  .withColumn('training_date', current_date() ) # 予測を実行した日付のカラムを追加
)

results.createOrReplaceTempView('new_forecasts')

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のクエリを発行することで、上記のモデリング・予測が実行されます。

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM new_forecasts

# COMMAND ----------

# MAGIC %md
# MAGIC 予測結果をクエリできるように、テーブルとして保存します。多数のモデル構築を行うため、クラスターのスペックに依存しますが、処理には数分の時間を要します。
# MAGIC 
# MAGIC また、ここでは`using delta`を指定することで、Delta Lake形式でテーブルを作成しています。[Delta Lake](https://databricks.com/product/delta-lake-on-databricks)を活用することで、ペタバイトオーダーのデータに対しても、高い信頼性、高速な検索性能を付与することが可能となります。

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists forecasts;
# MAGIC 
# MAGIC -- create forecast table
# MAGIC create table if not exists forecasts (
# MAGIC   date date,
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   sales float,
# MAGIC   sales_predicted float,
# MAGIC   sales_predicted_upper float,
# MAGIC   sales_predicted_lower float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC -- load data to it
# MAGIC insert into forecasts
# MAGIC select 
# MAGIC   ds as date,
# MAGIC   store,
# MAGIC   item,
# MAGIC   y as sales,
# MAGIC   yhat as sales_predicted,
# MAGIC   yhat_upper as sales_predicted_upper,
# MAGIC   yhat_lower as sales_predicted_lower,
# MAGIC   training_date
# MAGIC from new_forecasts;

# COMMAND ----------

# MAGIC %md
# MAGIC 1のモデルと同じようにモデルを評価する必要がありますが、<br>
# MAGIC ここでもpandas UDF を利用することで店舗・アイテムごとに評価指標を算出できます。

# COMMAND ----------

import pandas as pd

# 評価指標のカラム定義
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# 評価指標の算出
@pandas_udf( eval_schema, PandasUDFType.GROUPED_MAP )
def evaluate_forecast( evaluation_pd ):
  
  # データセットのストアとアイテムを取得
  training_date = evaluation_pd['training_date'].iloc[0]
  store = evaluation_pd['store'].iloc[0]
  item = evaluation_pd['item'].iloc[0]
  
  # 評価指標を算出
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # 結果を結合
  results = {'training_date':[training_date], 'store':[store], 'item':[item], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# COMMAND ----------

# calculate metrics
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # 評価を履歴データ(正解ラベル)がある期間に制限
    .select('training_date', 'store', 'item', 'y', 'yhat')
    .groupBy('training_date', 'store', 'item')
    .apply(evaluate_forecast)
    )
results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------

# MAGIC %md
# MAGIC 評価指標もクエリ可能なテーブルに永続化しておきます。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC drop table if exists forecast_evals;
# MAGIC 
# MAGIC create table if not exists forecast_evals (
# MAGIC   store integer,
# MAGIC   item integer,
# MAGIC   mae float,
# MAGIC   mse float,
# MAGIC   rmse float,
# MAGIC   training_date date
# MAGIC   )
# MAGIC using delta
# MAGIC partitioned by (training_date);
# MAGIC 
# MAGIC insert into forecast_evals
# MAGIC select
# MAGIC   store,
# MAGIC   item,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse,
# MAGIC   training_date
# MAGIC from new_forecast_evals;

# COMMAND ----------

# MAGIC %md
# MAGIC これで、ストアとアイテムの組み合わせごとに予測を作成し、それぞれの基本的な評価指標を生成できました。<br>
# MAGIC 対象を製品1に限定し、店舗1から店舗10それぞれの予測結果を見てみます。

# COMMAND ----------

# MAGIC %sql select count(*) from forecasts

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   date,
# MAGIC   sales_predicted,
# MAGIC   sales_predicted_upper,
# MAGIC   sales_predicted_lower
# MAGIC FROM forecasts a
# MAGIC WHERE item = 1 AND
# MAGIC       --store IN (1, 2, 3, 4, 5) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md
# MAGIC アイテム#1について、店舗ごとの予測の精度指標を見てみます。

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   store,
# MAGIC   mae,
# MAGIC   mse,
# MAGIC   rmse
# MAGIC FROM forecast_evals a
# MAGIC WHERE item = 1 AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md
# MAGIC 以下はモデル#1 (店舗とアイテムの組み合わせを考慮しない場合) の精度指標です。<br>
# MAGIC 全てのアイテムについて確認していないので断定はできませんが、細かい粒度での予測ができるようになりました。

# COMMAND ----------

print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md ## 参考情報
# MAGIC 
# MAGIC - [スターバックスにおける事例：Facebook Prophet と Azure Databricks を利用した大規模な需要予測 \- Databricks](https://databricks.com/jp/p/webinar/starbucks-forecast-demand-at-scale-facebook-prophet-azure-databricks)
# MAGIC - [Time Series Forecasting With Prophet And Spark \- Databricks](https://databricks.com/blog/2020/01/27/time-series-forecasting-prophet-spark.html)
# MAGIC - [Fine Grained Demand Forecasting \- Databricks](https://pages.databricks.com/rs/094-YMS-629/images/Fine-Grained-Time-Series-Forecasting.html)
