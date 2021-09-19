# Databricks notebook source
# MAGIC %md
# MAGIC # 時系列予測ライブラリ ProphetとSparkとの連携
# MAGIC 
# MAGIC 2021/9/15 
# MAGIC 
# MAGIC by Masahiko Kitamura
# MAGIC 
# MAGIC ## 1. 時系列予測とProphet
# MAGIC 
# MAGIC 時系列予測は周期性や季節性変動がある事象に対して予測を行います。例えば、ある商品の毎月の売り上げを考えると、商品の特性で夏に売り上が上がり、また、週末や休日前になると多く売れる、など様々な季節性、周期性要因が売り上げに関与してきます。時系列予測では、こうした季節性、周期性要因をうまくモデル化することが求められます。
# MAGIC 
# MAGIC [Prophet](https://facebook.github.io/prophet/)はこうした時系列予測のためのオープンソースライブラリです。Facebook社のCore Data Scienceチームが開発・リリースしており、年毎、週毎、日毎の周期性に加え、休日の影響などを考慮して非線形な傾向を持つ時系列データをシンプルにモデル化できるという特徴があります。さらに、異常値や欠損データの扱いにも強く、また、人間が理解しやすいパラメータやドメイン知識などを加えることでモデルの精度を向上させる機能も備えています。
# MAGIC 
# MAGIC PropehtはRおよびPythonで利用可能です。今回は、Pythonを使用したProphetの使用方法の概要とSparkと連携によりProphetをスケールさせる方法について見ていきます。
# MAGIC 
# MAGIC ## 2. Prophetの実行例と拡張性
# MAGIC 
# MAGIC Prophetの第一の特徴はシンプルに使用できる点にあります。Pythonの機械学習ライブラリでよく用いられるScikit-learnのモデル化の手順と同等になるような関数設計になっており、`fit()`でモデルを作成し、`predict()`で予測(スコアリング)する仕様になっています。
# MAGIC 
# MAGIC モデル作成において、入力として使用する学習データとしては単純に以下の2つのカラムを持つDataFrameを用意します。
# MAGIC 
# MAGIC * `ds` : 日時もしくはタイムスタンプ
# MAGIC * `y` : 予測を実施する数値 (例: 商品の売り上げ数)
# MAGIC 
# MAGIC ここでは、Prophetのgitリポジトリに含まれるサンプルデータを使って実際のコードを見ていきます。
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## 3. Databricks上でのProphetの利用
# MAGIC 
# MAGIC 
# MAGIC ## 4. ProphetをSpark上でスケールさせる

# COMMAND ----------

# MAGIC %sh pip install pystan==2.19.1.1 && pip install  prophet

# COMMAND ----------

# MAGIC %sh curl -O 'https://raw.githubusercontent.com/facebook/prophet/master/examples/example_wp_log_peyton_manning.csv'

# COMMAND ----------

import pandas as pd
from prophet import Prophet

df = pd.read_csv('example_wp_log_peyton_manning.csv')

# COMMAND ----------

print( df.head() )

# COMMAND ----------

display(df)

# COMMAND ----------

m = Prophet() # インスタンス作成
m.fit(df) # モデルの学習
future = m.make_future_dataframe(periods=365) # 1年先までの日時がdsカラムに入っているデータフレームを用意する(予測データを入れるハコを作る)
forecast = m.predict(future) # 予測する

fig1 = m.plot(forecast, figsize=(20, 12)) # 結果のプロット#1
fig2 = m.plot_components(forecast) # 結果のプロット#2

# COMMAND ----------

# MAGIC %md
# MAGIC どう言った要因を加えることができるのかの概要を見ていく。
# MAGIC 詳細やここで触れていない拡張性につていは[公式のドキュメント](https://facebook.github.io/prophet/docs/)を参照ください。
# MAGIC 
# MAGIC * 休日の要素を加える
# MAGIC * カスタムの周期性季節性を加える
# MAGIC * 他の要因に左右される季節性を加える
# MAGIC * 信頼性区間の変更
# MAGIC * Hyper param tuning
# MAGIC 
# MAGIC * 

# COMMAND ----------

# Python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
