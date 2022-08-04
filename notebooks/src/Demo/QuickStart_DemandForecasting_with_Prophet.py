# Databricks notebook source
# MAGIC %md # 売上の需要予測と可視化
# MAGIC 
# MAGIC * データ: KaggleのInstacart
# MAGIC * 需要予測ライブラリ: Prophet
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

table_name = 'mk1112'

spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name}')

(
  spark.createDataFrame(forecast_pd)
  .write
  .mode('overwrite')
  .saveAsTable(f'{database_name}.{table_name}') # マネージドテーブルで保存(システム側でデータの保存先を自動で管理する方式。ユーザーがデータ保存先パスを指定しなくていい)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC --永続化したデータを確認する
# MAGIC SELECT * FROM handson20220810.<上で自身がつけたテーブル名>
# MAGIC -- 例) SELECT * FROM handson20220810.mk1112

# COMMAND ----------

# MAGIC %md ## 4. 予測を可視化する

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## 99. 環境のクリーンアップ

# COMMAND ----------

# MAGIC %sql
# MAGIC -- adminの方が1度実行すればクリーンアップできます。
# MAGIC DROP DATABASE handson20220810 CASCADE

# COMMAND ----------


