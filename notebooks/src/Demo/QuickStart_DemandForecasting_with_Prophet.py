# Databricks notebook source
# MAGIC %md # 売上の需要予測と可視化
# MAGIC 
# MAGIC * データ: [Kaggleの"Store Item Demand Forecasting Challenge"](https://www.kaggle.com/c/demand-forecasting-kernels-only/data?select=train.csv)のデータを使用
# MAGIC * 需要予測ライブラリ: [Prophet](https://qiita.com/ktmrmshk/items/79520d5beed1787f595e)
# MAGIC * 流れ
# MAGIC   1. データを準備する
# MAGIC   1. データの感触をつかむ (探索・EDA・簡単に可視化)
# MAGIC   1. 需要予測を実施する
# MAGIC      * 時系列データの予測モデルをProphetで作成
# MAGIC      * 予測モデルから未来の需要を予測する
# MAGIC      * データを保存する(DeltaLake形式でテーブル化)
# MAGIC   1. 予測を可視化する
# MAGIC      * (Notebook上)
# MAGIC      * Databricks SQL(BIツール上)

# COMMAND ----------

# MAGIC %md ## 1. データを準備する

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col

# ひとまずpandasデータフレームで読み込む(pandasだとURLから読み込める)
p_df_train = pd.read_csv('https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/example/train.csv.zip')


# PandasデータフレームからSparkのデータフレームにする
# メリット:
#   * Sparkの分散処理環境で処理される
#   * SQLとPythonを相互に使える
#   * データの永続化がラク (DeltaLakeで(RDBMSサーバなしで)データウェアハウス機能が使えデータの永続化が簡単)

s_df_train = (
  spark.createDataFrame(p_df_train)
  .withColumn('dte', col('date').cast('date'))
  .drop('date')
  .withColumnRenamed('dte', 'date')
  .select('date', 'store', 'item', 'sales')
)

# SQLからも参照したいので、temp viewを作成する
# (このNotebook上に限り、このtemp view名でSQLが使える)
s_df_train.createOrReplaceTempView('train')

# COMMAND ----------

# データを表示してみる(Python)
display(s_df_train)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- データを表示してみる(SQL) -- データ実体は同じなので、表示結果も同じ
# MAGIC SELECT * FROM train

# COMMAND ----------

# MAGIC %md ## 2.データの感触をつかむ (探索・EDA・簡単に可視化)

# COMMAND ----------

# MAGIC %sql
# MAGIC --単純に線グラフ
# MAGIC SELECT * FROM train

# COMMAND ----------

# MAGIC %sql
# MAGIC --年次推移
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %sql
# MAGIC --月次推移
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 週次推移
# MAGIC SELECT
# MAGIC   YEAR(date) as year,
# MAGIC   (
# MAGIC     CASE
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sun' THEN 0
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Mon' THEN 1
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Tue' THEN 2
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Wed' THEN 3
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Thu' THEN 4
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Fri' THEN 5
# MAGIC       WHEN DATE_FORMAT(date, 'E') = 'Sat' THEN 6
# MAGIC     END
# MAGIC   ) % 7 as weekday,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     date,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM train
# MAGIC   GROUP BY date
# MAGIC  ) x
# MAGIC GROUP BY year, weekday
# MAGIC ORDER BY year, weekday;

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
# MAGIC   FROM train
# MAGIC   GROUP BY TRUNC(date, 'MM'), item
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), item
# MAGIC ORDER BY month, item

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
# MAGIC   FROM train
# MAGIC   GROUP BY TRUNC(date, 'MM'), store
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), store
# MAGIC ORDER BY month, store;

# COMMAND ----------

# MAGIC %md ## 3.需要予測を実施する

# COMMAND ----------

# MAGIC %md #### 時系列データの予測モデルをProphetで作成

# COMMAND ----------

# 時系列データの予測モデルをProphetで作成
# 
# 「店舗:4」で売っている「商品:7」の売り上げを予想するモデルを作る
# Prophetライブラリでモデルを作るには「時刻、その時の数値」の２つのカラムを持つテーブル(pandasデータフレーム)を用意する。

sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    sales as y
  FROM train
  WHERE store=4 AND item=7
  ORDER BY ds
  '''

# 上記のSQLを実行して、それをpandasデータフレームに変換する
history_pd = spark.sql(sql_statement).toPandas()

# Nullになっているレコードを除く
history_pd = history_pd.dropna()

# 整えたpandasデータフレームを確認
display(history_pd)

# COMMAND ----------

# 続けて上記のデータで予測モデルを作成(モデルトレーニング)
from prophet import Prophet
import logging
# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)


model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# fit()を呼び出す！
model.fit(history_pd)


# COMMAND ----------

# MAGIC %md ####予測モデルから未来の需要を予測する

# COMMAND ----------

# 学習したモデルを使って90日先まで推定する

## 90日先まで、pandasデータフレームの枠を作っておく
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# predict()関数! で予測する
forecast_pd = model.predict(future_pd)

# 予測結果の確認！
display(forecast_pd)

# COMMAND ----------

# 予測モデルのトレンドを見る(Prophetの機能)
trends_fig = model.plot_components(forecast_pd)
# display(trends_fig)

# COMMAND ----------

# 過去データと予測データを比較する (Prophetの機能)
predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')

# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

#display(predict_fig)

# COMMAND ----------

# MAGIC %md **注** この可視化は色々含まれていて一見複雑に見えます。Bartosz Mikulskiが[優れた内訳](https://www.mikulskibartosz.name/prophet-plot-explained/)を提供しており、それをチェックする価値は十分にあります。 簡単に言うと、黒い点は我々の実績を、濃い青色の線は我々の予測を、薄い青色の帯は我々の（95％）不確実性区間を表しています。

# COMMAND ----------

# どのくらい予測が合っているのかをトレーニング期間のデータで誤差を評価する
# 二乗誤差平均(MSE)、絶対誤差平均(MAE)、二乗誤差平均のルート(RMSE)

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date

# get historical actuals & predictions for comparison
actuals_pd = history_pd[ history_pd['ds'] < date(2018, 1, 1) ]['y']
predicted_pd = forecast_pd[ forecast_pd['ds'] < pd.to_datetime('2018-01-01') ]['yhat']

# calculate evaluation metrics
mae = mean_absolute_error(actuals_pd, predicted_pd)
mse = mean_squared_error(actuals_pd, predicted_pd)
rmse = sqrt(mse)

# print metrics to the screen
print( '\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse) )

# COMMAND ----------

# MAGIC %md #### データを保存する(DeltaLake形式でテーブル化・永続化)

# COMMAND ----------

database_name = 'handson20220810'
table_name = 'ここを自分の名前などのユニークな文字列に置き換えてください!'
# 例 table_name = 'mk1112'

spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name}')

# 「予測結果」をテーブルとして保存
(
  spark.createDataFrame(forecast_pd)
  .write
  .mode('overwrite')
  .saveAsTable(f'{database_name}.{table_name}_predicted') # マネージドテーブルで保存(システム側でデータの保存先を自動で管理する方式。ユーザーがデータ保存先パスを指定しなくていい)
)


# 「もとの売り上げデータ」もテーブルとして保存(予測と実績値を比較するのに使用)
(
  s_df_train
  .write
  .mode('overwrite')
  .saveAsTable(f'{database_name}.{table_name}_history')
)

# COMMAND ----------

# MAGIC %sql
# MAGIC --永続化したデータを確認する
# MAGIC SELECT * FROM handson20220810.<上で自身がつけたテーブル名>_predicted
# MAGIC -- 例) SELECT * FROM handson20220810.mk1112_predicted

# COMMAND ----------

# MAGIC %md ## 4. 予測を可視化する

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 以下のようなダッシュボードを作成します。
# MAGIC 
# MAGIC ![dashboard](https://sajpstorage.blob.core.windows.net/demo-asset-workshop2021/example/dashbaord_example.png)
# MAGIC 
# MAGIC ##### 1. 左メニューの一番上のスイッチUIから「SQL」を選択してDatabricks SQLへ移動する。
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ##### 2. データカタログ上でテーブル確認 
# MAGIC ###### 2.1 左メニューの「データ」から上記で作成したテーブル(`hive_metastore > handson20220810 > 自分ユニークな名前のテーブル`)を確認する。
# MAGIC 
# MAGIC * UIからスキーマ、サンプルデータなどを確認
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ##### 3. 売り上げ予測のグラグを作る
# MAGIC ###### 3.1　. 左メニューから「SQLエディタ」を開いて、以下のSQLをコピー&ペーストし、「`{自分のユニーク文字列}_売り上げ予想`」という名前で保存。
# MAGIC 
# MAGIC ```
# MAGIC SELECT
# MAGIC   *
# MAGIC FROM
# MAGIC   handson20220810.mk1112_predicted
# MAGIC ```
# MAGIC 
# MAGIC ###### 3.2 上記クエリを実行して、テーブル結果を表示。その後、テーブル結果の右上の「+ビジュアライゼーションを追加」で、Lineチャートを作成。結果タブからチャートの名前「`売り上げ予想`」に変更する。
# MAGIC 
# MAGIC * x軸(横軸): `ds`
# MAGIC * y軸(縦軸): `yhat`, `yhat_upper`, `yhat_lower`
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ##### 4. 実績値のグラグを作る
# MAGIC ###### 4.1 左メニューから「SQLエディタ」を開いて、以下のSQLをコピー&ペーストし、「`{自分のユニーク文字列}_アイテム毎・月毎のトレンド`」という名前で保存。
# MAGIC 
# MAGIC ```
# MAGIC SELECT -- アイテム毎で、月売上の平均を算出
# MAGIC   MONTH(month) as month,
# MAGIC   item,
# MAGIC   AVG(sales) as sales
# MAGIC FROM (
# MAGIC   SELECT -- 月ごとの売り上げで集計したテーブルをサブクエリに
# MAGIC     TRUNC(date, 'MM') as month,
# MAGIC     item,
# MAGIC     SUM(sales) as sales
# MAGIC   FROM handson20220810.mk1112_history
# MAGIC   GROUP BY TRUNC(date, 'MM'), item
# MAGIC   ) x
# MAGIC GROUP BY MONTH(month), item
# MAGIC ORDER BY month, item
# MAGIC ```
# MAGIC 
# MAGIC ###### 4.2 step4, 5と同様に適宜名前を置き換えて実施する
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ##### 5. 複数のグラフを一つのダッシュボードにまとめる
# MAGIC ###### 5.1 左メニューから「ダッシュボード」を選択し、その後「ダッシュボードを作成」から「`{自分のユニーク文字列}_ダッシュボード`」という名前で作成する
# MAGIC ###### 5.2 ダッシュボード(編集モード)の上部「追加 > 可視化」から上記で作成した2つのグラフを追加・配置する
# MAGIC ###### 5.3 「編集完了」ボタンで固定化できる。他のグラフを追加する場合は、再度「編集」ボタンを押す

# COMMAND ----------

# MAGIC %md ## 99. 環境のクリーンアップ

# COMMAND ----------

# MAGIC %sql
# MAGIC -- adminの方が1度実行すればクリーンアップできます。
# MAGIC -- (全参加者でデータベース名を同じにしているので)
# MAGIC DROP DATABASE IF EXISTS handson20220810 CASCADE
