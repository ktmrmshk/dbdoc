# Databricks notebook source
# MAGIC %md このノートブックの目的は、Databricksの分散計算能力を活用し、効率的にストアアイテムレベルで多数のきめ細かい予測を生成する方法を説明することです。 このノートブックは、以前Spark 2.x用に開発されたノートブックをSpark 3.x用に更新したものです。 このノートブックの **UPDATE** マークは、Spark 3.x または Databricks プラットフォームの新機能を反映するためのコードの変更を示しています。

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC この演習では、需要予測のためのますます人気のあるライブラリである [FBProphet](https://facebook.github.io/prophet/) を使用します。このライブラリは、Databricks 7.1 以降を実行しているクラスタのノートブックセッションにロードされます。
# MAGIC 
# MAGIC **UPDATE** Databricks 7.1では、%pipマジックコマンドを使用して[notebook-scoped libraries](https://docs.databricks.com/dev-tools/databricks-utils.html#library-utilities)をインストールすることができるようになりました。

# COMMAND ----------

# DBTITLE 1,必要なライブラリのインストール
# MAGIC %pip install pystan==2.19.1.1  # per https://github.com/facebook/prophet/commit/82f3399409b7646c49280688f59e5a3d2c936d39#comments
# MAGIC %pip install fbprophet==0.6

# COMMAND ----------

# MAGIC %md ## ステップ1：データを調べる
# MAGIC 
# MAGIC 学習用データセットとして、10店舗50アイテムの5年間の店舗・アイテム単位販売データを利用する。 このデータセットは過去のKaggleコンペティションの一部として公開されており、[こちら](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)からダウンロードすることが可能です。
# MAGIC 
# MAGIC ダウンロード後、*train.csv.zip*ファイルを解凍し、解凍したCSVを*/FileStore/demand_forecast/train/*　にアップロードすることが可能です( [ドキュメント](https://docs.databricks.com/data/databricks-file-system.html#!#user-interface))。Databricksでデータセットにアクセスできるようになったので、モデリング準備のためにデータセットを探索することができます。

# COMMAND ----------

# DBTITLE 1,データセットへのアクセス
from pyspark.sql.types import *

# structure of the training data set
train_schema = StructType([
  StructField('date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('sales', IntegerType())
  ])

# read the training file into a dataframe
train = spark.read.csv(
  'dbfs:/FileStore/demand_forecast/train/train.csv', 
  header=True, 
  schema=train_schema
  )

# make the dataframe queriable as a temporary view
train.createOrReplaceTempView('train')

# show data
display(train)

# COMMAND ----------

# MAGIC %md 需要予測を行う際、一般的な傾向や季節性に関心を持つことがよくあります。 まず、販売台数の年次推移を調べてみましょう。

# COMMAND ----------

# DBTITLE 1,年次推移
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   year(date) as year, 
# MAGIC   sum(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY year(date)
# MAGIC ORDER BY year;

# COMMAND ----------

# MAGIC %md このデータから、各店舗の総販売台数が概して増加傾向にあることは明らかです。もし、これらの店舗が展開する市場についてもっと詳しい知識があれば、予測期間中に近づくと思われる最大の成長力があるかどうかを見極めたいと思うかもしれません。 しかし、そのような知識がなく、このデータセットをざっと見ただけでは、数日後、数ヶ月後、あるいは1年後の予測をするのが目的であれば、その期間中、直線的な成長が続くと想定してもよさそうです。
# MAGIC 
# MAGIC 次に、季節性について見てみましょう。 各年の各月のデータを集計すると、1年ごとに明確な季節パターンが観察され、売上高の全体的な伸びとともに規模が拡大しているように見えます。

# COMMAND ----------

# DBTITLE 1,月次推移
# MAGIC %sql
# MAGIC 
# MAGIC SELECT 
# MAGIC   TRUNC(date, 'MM') as month,
# MAGIC   SUM(sales) as sales
# MAGIC FROM train
# MAGIC GROUP BY TRUNC(date, 'MM')
# MAGIC ORDER BY month;

# COMMAND ----------

# MAGIC %md 週次で集計すると、日曜日（平日0日）をピークに、月曜日（平日1日）に大きく落ち込み、その後、1週間かけて着実に上昇し、日曜日の高値に戻るという季節パターンが顕著に観察される。 このパターンは、5年間の観測でかなり安定しているようです。
# MAGIC 
# MAGIC **UPDATE** Spark 3 の [Proleptic Gregorian calendar](https://databricks.com/blog/2020/07/22/a-comprehensive-look-at-dates-and-timestamps-in-apache-spark-3-0.html) への移行に伴い、CAST(DATE_FORMAT(date, 'u')) の 'u' オプションが削除されました。現在では、'E'を使用して同様の出力を提供しています。

# COMMAND ----------

# DBTITLE 1,週次推移
# MAGIC %sql
# MAGIC 
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

# MAGIC %md さて、データ内の基本的なパターンを把握したところで、予測をどのように構築するか探ってみましょう。

# COMMAND ----------

# MAGIC %md ## ステップ2：1つの予測を立てる
# MAGIC 
# MAGIC 店舗と商品の個々の組み合わせについて予測を作成する前に、FBProphetの使い方を理解するために、1つの予測を作成することが役に立つかもしれません。
# MAGIC 
# MAGIC 最初のステップは、モデルを訓練するための過去のデータセットを組み立てることです。

# COMMAND ----------

# DBTITLE 1,1つの「商品 x 店舗」の組み合わせのデータを取得する
# query to aggregate data to date (ds) level
sql_statement = '''
  SELECT
    CAST(date as date) as ds,
    sales as y
  FROM train
  WHERE store=1 AND item=1
  ORDER BY ds
  '''

# assemble dataset in Pandas dataframe
history_pd = spark.sql(sql_statement).toPandas()

# drop any missing records
history_pd = history_pd.dropna()

# COMMAND ----------

# MAGIC %md さて、fbprophet ライブラリをインポートしますが、使用時に少し冗長になることがあるので、環境のロギング設定を微調整する必要があります。

# COMMAND ----------

# DBTITLE 1,Prophetライブラリのimport
from fbprophet import Prophet
import logging

# disable informational messages from fbprophet
logging.getLogger('py4j').setLevel(logging.ERROR)

# COMMAND ----------

# MAGIC %md データのレビューによると、全体的な成長パターンを線形に設定し、週単位と年単位の季節性パターンを評価できるようにする必要があるようです。また、季節パターンは売上高の全体的な成長とともに増加するようなので、季節性モードを乗算に設定するのもよいでしょう。

# COMMAND ----------

# DBTITLE 1,Prophet Modelをトレーニング
# set model parameters
model = Prophet(
  interval_width=0.95,
  growth='linear',
  daily_seasonality=False,
  weekly_seasonality=True,
  yearly_seasonality=True,
  seasonality_mode='multiplicative'
  )

# fit the model to historical data
model.fit(history_pd)

# COMMAND ----------

# MAGIC %md さて、学習済みモデルができたので、それを使って90日予報を作成してみましょう。

# COMMAND ----------

# DBTITLE 1,学習したモデルを使って90日先まで推定する
# define a dataset including both historical dates & 90-days beyond the last available date
future_pd = model.make_future_dataframe(
  periods=90, 
  freq='d', 
  include_history=True
  )

# predict over the dataset
forecast_pd = model.predict(future_pd)

display(forecast_pd)

# COMMAND ----------

# MAGIC %md 私たちのモデルの性能はどうだったでしょうか？ここでは、私たちのモデルの一般的な傾向と季節的な傾向をグラフで見ることができます。

# COMMAND ----------

# DBTITLE 1,Examine Forecast Components
trends_fig = model.plot_components(forecast_pd)
display(trends_fig)

# COMMAND ----------

# MAGIC %md 
# MAGIC ここでは、グラフを見やすくするために、過去1年分のデータに限定していますが、実際のデータと予測データがどのように一致しているか、また、将来の予測も見ることができます。
# MAGIC ここでは、グラフを見やすくするために、過去1年分のデータに限定していますが、実際のデータと予測データがどのように一致しているか、また、将来の予測も見ることができます。

# COMMAND ----------

# DBTITLE 1,過去データと予測データを比較する
predict_fig = model.plot( forecast_pd, xlabel='date', ylabel='sales')

# adjust figure to display dates from last year + the 90 day forecast
xlim = predict_fig.axes[0].get_xlim()
new_xlim = ( xlim[1]-(180.0+365.0), xlim[1]-90.0)
predict_fig.axes[0].set_xlim(new_xlim)

display(predict_fig)

# COMMAND ----------

# MAGIC %md **注** この可視化は色々含まれていて一見複雑に見えます。Bartosz Mikulskiが[優れた内訳](https://www.mikulskibartosz.name/prophet-plot-explained/)を提供しており、それをチェックする価値は十分にあります。 簡単に言うと、黒い点は我々の実績を、濃い青色の線は我々の予測を、薄い青色の帯は我々の（95％）不確実性区間を表しています。

# COMMAND ----------

# MAGIC %md 目視検査は有用ですが、予測を評価するより良い方法は、セット内の実際の値に対する予測値の平均絶対誤差、平均二乗誤差、ルート平均二乗誤差の値を計算することです。
# MAGIC 
# MAGIC **UPDATE** pandasの機能の変更により、日付文字列を正しいデータ型に変換するために*pd.to_datetime*を使用する必要があります。

# COMMAND ----------

# DBTITLE 1,評価指標の算出
import pandas as pd
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

# MAGIC %md FBProphetは、あなたの予測が時間とともにどのように持ちこたえるか評価するための[追加手段](https://facebook.github.io/prophet/docs/diagnostics.html)を提供します。予測モデルを構築するときに、これらや追加のテクニックを使うことを強くお勧めしますが、ここではスケーリングの課題に焦点を当てるため、これを省きます。

# COMMAND ----------

# MAGIC %md ## ステップ3：需要予測の生成をスケールさせる
# MAGIC 
# MAGIC このような仕組みが出来上がったところで、当初の目標であった、個々の店舗と商品の組み合わせに対する多数の細かいモデルと予測を構築することに取り組みます。 まず、店舗・商品・日付の粒度の売上データを集めることから始めます。
# MAGIC 
# MAGIC **注**: このデータセットのデータは、すでにこの粒度で集約されているはずですが、期待通りのデータ構造を確保するために明示的に集約しています。

# COMMAND ----------

# DBTITLE 1,Retrieve Data for All Store-Item Combinations
sql_statement = '''
  SELECT
    store,
    item,
    CAST(date as date) as ds,
    SUM(sales) as y
  FROM train
  GROUP BY store, item, ds
  ORDER BY store, item, ds
  '''

store_item_history = (
  spark
    .sql( sql_statement )
    .repartition(sc.defaultParallelism, ['store', 'item'])
  ).cache()

# COMMAND ----------

# MAGIC %md 店舗・アイテム・日付のレベルで集約されたデータを、どのようにFBProphetに渡すかを検討する必要があります。もし我々のゴールが各店舗と商品の組み合わせのモデルを構築することであるなら、先ほど組み立てたデータセットから店舗商品のサブセットを渡して、そのサブセットでモデルをトレーニングし、店舗商品予測を受け取る必要があります。予測はこのような構造のデータセットとして返されることを期待します。ここでは、予測が組み立てられた店舗とアイテムの識別子を保持し、出力をProphetモデルによって生成されたフィールドの関連するサブセットだけに限定しています。

# COMMAND ----------

# DBTITLE 1,予測出力のスキーマを定義する
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

# MAGIC %md モデルをトレーニングして予測を生成するために、Pandasの関数を活用します。 この関数は店舗と商品の組み合わせで構成されるデータのサブセットを受け取るように定義します。 この関数は、前のセルで特定されたフォーマットで予測を返します。
# MAGIC 
# MAGIC **UPDATE** Spark 3.0では、pandas関数がpandas UDFにあった機能を置き換えます。 非推奨のpandas UDF構文はまだサポートされていますが、時間の経過とともに段階的に廃止される予定です。 新しい、合理化されたpandas functions APIの詳細については、[このドキュメント](https://databricks.com/blog/2020/05/20/new-pandas-udfs-and-python-type-hints-in-the-upcoming-release-of-apache-spark-3-0.html)を参照してください。

# COMMAND ----------

# DBTITLE 1,モデルの学習と予測のための関数を定義する
def forecast_store_item( history_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # TRAIN MODEL AS BEFORE
  # --------------------------------------
  # remove missing values (more likely at day-store-item level)
  history_pd = history_pd.dropna()
  
  # configure the model
  model = Prophet(
    interval_width=0.95,
    growth='linear',
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative'
    )
  
  # train the model
  model.fit( history_pd )
  # --------------------------------------
  
  # BUILD FORECAST AS BEFORE
  # --------------------------------------
  # make predictions
  future_pd = model.make_future_dataframe(
    periods=90, 
    freq='d', 
    include_history=True
    )
  forecast_pd = model.predict( future_pd )  
  # --------------------------------------
  
  # ASSEMBLE EXPECTED RESULT SET
  # --------------------------------------
  # get relevant fields from forecast
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  
  # get relevant fields from history
  h_pd = history_pd[['ds','store','item','y']].set_index('ds')
  
  # join history and forecast
  results_pd = f_pd.join( h_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  
  # get store & item from incoming data set
  results_pd['store'] = history_pd['store'].iloc[0]
  results_pd['item'] = history_pd['item'].iloc[0]
  # --------------------------------------
  
  # return expected dataset
  return results_pd[ ['ds', 'store', 'item', 'y', 'yhat', 'yhat_upper', 'yhat_lower'] ]  

# COMMAND ----------

# MAGIC %md この関数の中では多くのことが行われていますが、モデルの学習と予測が行われる最初の2つのブロックのコードをこのノートブックの前の部分のセルと比較すると、コードは以前とほとんど同じであることがわかります。本当に新しいコードが導入されているのは、必要な結果セットの組み立てだけで、それはかなり標準的なPandasのデータフレーム操作で構成されています。

# COMMAND ----------

# MAGIC %md では、pandasの関数を呼び出して、予測を作ってみましょう。 これは、店舗とアイテムの履歴データセットをグループ化することによって行います。 そして、各グループに関数を適用し、データ管理のために今日の日付を *training_date* として追加します。
# MAGIC 
# MAGIC **UPDATE** 前回のアップデートで、pandas UDFの代わりにapplyInPandas()を使ってpandas関数を呼び出しています。

# COMMAND ----------

# DBTITLE 1,各(店舗 x 商品)の組み合わせに予測関数を適用する
from pyspark.sql.functions import current_date

results = (
  store_item_history
    .groupBy('store', 'item')
      .applyInPandas(forecast_store_item, schema=result_schema)
    .withColumn('training_date', current_date() )
    )

results.createOrReplaceTempView('new_forecasts')

display(results)

# COMMAND ----------

# MAGIC %md 予測結果を報告したいので、照会可能なテーブル構造で保存しておきましょう。

# COMMAND ----------

# DBTITLE 1,予測結果の永続化(Deltaテーブル)
# MAGIC %sql
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

# MAGIC %md しかし、それぞれの予測はどの程度良い（あるいは悪い）のでしょうか？ pandasの関数テクニックを使って、以下のように各店舗商品の予測に対する評価指標を生成することができます。

# COMMAND ----------

# DBTITLE 1,同じ手法で各予測を評価する
# schema of expected result set
eval_schema =StructType([
  StructField('training_date', DateType()),
  StructField('store', IntegerType()),
  StructField('item', IntegerType()),
  StructField('mae', FloatType()),
  StructField('mse', FloatType()),
  StructField('rmse', FloatType())
  ])

# define function to calculate metrics
def evaluate_forecast( evaluation_pd: pd.DataFrame ) -> pd.DataFrame:
  
  # get store & item in incoming data set
  training_date = evaluation_pd['training_date'].iloc[0]
  store = evaluation_pd['store'].iloc[0]
  item = evaluation_pd['item'].iloc[0]
  
  # calulate evaluation metrics
  mae = mean_absolute_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  mse = mean_squared_error( evaluation_pd['y'], evaluation_pd['yhat'] )
  rmse = sqrt( mse )
  
  # assemble result set
  results = {'training_date':[training_date], 'store':[store], 'item':[item], 'mae':[mae], 'mse':[mse], 'rmse':[rmse]}
  return pd.DataFrame.from_dict( results )

# calculate metrics
results = (
  spark
    .table('new_forecasts')
    .filter('ds < \'2018-01-01\'') # limit evaluation to periods where we have historical data
    .select('training_date', 'store', 'item', 'y', 'yhat')
    .groupBy('training_date', 'store', 'item')
    .applyInPandas(evaluate_forecast, schema=eval_schema)
    )

results.createOrReplaceTempView('new_forecast_evals')

# COMMAND ----------

# MAGIC %md もう一度言いますが、各予測のメトリクスを報告したいと思うでしょうから、これらを照会可能なテーブルに永続化します。

# COMMAND ----------

# DBTITLE 1,評価指標の永続化(Deltaテーブル)
# MAGIC %sql
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

# MAGIC %md これで各店舗と商品の組み合わせの予測を作成し、それぞれについて基本的な評価指標を作成しました。 この予測データを見るために、簡単なクエリーを発行することができます（ここでは1～3店舗にわたる商品1に限定しています）。

# COMMAND ----------

# DBTITLE 1,予測値を可視化する
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
# MAGIC       store IN (1, 2, 3) AND
# MAGIC       date >= '2018-01-01' AND
# MAGIC       training_date=current_date()
# MAGIC ORDER BY store

# COMMAND ----------

# MAGIC %md そして、これらのそれぞれについて、それぞれの予測の信頼性を評価するのに役立つ指標を取り出すことができます。

# COMMAND ----------

# DBTITLE 1,評価指標を取得する
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
