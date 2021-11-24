# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## 足を運ぶ人を予測するためのキャンペーン効果の識別 - ETL
# MAGIC ###### 広告業界では、広告費がどのような結果をもたらしたかという情報をクライアントに提供することが最も重要な業務の一つです。また、アトリビュートされた情報をクライアントに迅速に提供できれば、それに越したことはありません。オフラインの活動と広告キャンペーンで提供されたインプレッションを結びつけるために、企業はアトリビューションを行う必要があります。アトリビューションにはかなりのコストがかかります。また、常に更新されるデータセットに対してアトリビューションを実行することは、適切な技術がなければ困難です。
# MAGIC 
# MAGIC ###### 幸いなことに、Databricks社はUnified Data Analytics PlatformとDeltaにより、この作業を容易にしています。
# MAGIC #### このノートブックで行う主な手順は以下の通りです。 
# MAGIC 
# MAGIC * [SafeGraphフォーマット](https://www.safegraph.com/points-of-interest-poi-data-guide)の月次足動量時系列データの取り込み（モック） - ここでは、スキーマ（ブロンズ）に合わせてデータをモックしています。
# MAGIC * 月間時系列データに変換する - 日付ごとの閲覧数を数値化する（行＝日付）（シルバー）
# MAGIC * Subway RestaurantをNYCエリアに限定する（Gold）
# MAGIC * 機能の探索的な分析：分布のチェック、変数のトランスフォーム（ゴールド）
# MAGIC 
# MAGIC SafeGraphデータの詳細は以下の通りです。
# MAGIC 
# MAGIC #### SafeGraph Patternsとは？
# MAGIC * SafeGraph の Places Patterns は、匿名化された訪問者の足取りと訪問者の人口統計データを集約したデータセットで、米国内の約 360 万の POI（Point of Interest）で利用可能です。
# MAGIC * ここでは、リミテッド・サービス・レストランの来店セットの履歴データ（2019年1月～2020年2月）を見てみましょう。

# COMMAND ----------

displayHTML('''<img src="https://databricks.com/wp-content/uploads/2020/10/mm-ref-arch-1.png" style="width:1100px;height:550px;">''')

# COMMAND ----------


dates=spark.sql("""select '201901' year_month, cast(cast('2019-01-01' as timestamp) as long) date_range_start, cast(cast('2019-01-31' as timestamp) as long) date_range_end
union all 
select '201902' year_month, cast(cast('2019-02-01' as timestamp) as long) date_range_start, cast(cast('2019-02-28' as timestamp) as long) date_range_end
union all
select '201903' year_month, cast(cast('2019-03-01' as timestamp) as long) date_range_start, cast(cast('2019-03-31' as timestamp) as long) date_range_end
union all
select '201904' year_month, cast(cast('2019-04-01' as timestamp) as long) date_range_start, cast(cast('2019-04-30' as timestamp) as long) date_range_end
union all
select '201905' year_month, cast(cast('2019-05-01' as timestamp) as long) date_range_start, cast(cast('2019-05-31' as timestamp) as long) date_range_end
union all
select '201906' year_month, cast(cast('2019-06-01' as timestamp) as long) date_range_start, cast(cast('2019-06-30' as timestamp) as long) date_range_end
union all
select '201907' year_month, cast(cast('2019-07-01' as timestamp) as long) date_range_start, cast(cast('2019-07-31' as timestamp) as long) date_range_end
union all
select '201908' year_month, cast(cast('2019-08-01' as timestamp) as long) date_range_start, cast(cast('2019-08-31' as timestamp) as long) date_range_end
union all
select '201909' year_month, cast(cast('2019-09-01' as timestamp) as long) date_range_start, cast(cast('2019-09-30' as timestamp) as long) date_range_end
union all
select '201910' year_month, cast(cast('2019-10-01' as timestamp) as long) date_range_start, cast(cast('2019-10-31' as timestamp) as long) date_range_end
union all
select '201911' year_month, cast(cast('2019-11-01' as timestamp) as long) date_range_start, cast(cast('2019-11-30' as timestamp) as long) date_range_end
union all
select '201912' year_month, cast(cast('2019-12-01' as timestamp) as long) date_range_start, cast(cast('2019-12-31' as timestamp) as long) date_range_end
union all
select '202001' year_month, cast(cast('2020-01-01' as timestamp) as long) date_range_start, cast(cast('2020-01-31' as timestamp) as long) date_range_end
union all
select '202002' year_month, cast(cast('2020-02-01' as timestamp) as long) date_range_start, cast(cast('2020-02-29' as timestamp) as long) date_range_end""")

# COMMAND ----------

# MAGIC %sh wget http://databricks.com/notebooks/safegraph_patterns_simulated__1_-91d51.csv

# COMMAND ----------

# MAGIC %fs 
# MAGIC cp file:/databricks/driver/safegraph_patterns_simulated__1_-91d51.csv /tmp/altdata_poi/foot_traffic.csv

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 生のフットトラフィックのインジェスト (ブロンズ)

# COMMAND ----------

# DBTITLE 1,フットトラフィックデータのCSV読み込み
raw_sim_ft = spark.read.format("csv").option("header", "true").option("sep", ",").load("/tmp/altdata_poi/foot_traffic.csv")

raw_sim_ft.createOrReplaceTempView("safegraph_sim_foot_traffic")

# COMMAND ----------

# MAGIC %sql select * from safegraph_sim_foot_traffic

# COMMAND ----------

# DBTITLE 1,ブロンズテーブルのデルタ化(永続化- Deltaフォーマットで書き込む)
raw_sim_ft.write.format('delta').mode('overwrite').save('/home/layla/data/table/footTrafficBronze')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 異なるMSAエリアでの店舗訪問の理解 （シルバー）
# MAGIC 
# MAGIC ###### 特徴量エンジニアリング: 注意 `Visit_by_date`はArrayで、データを別の行に分解する必要があります。

# COMMAND ----------

# DBTITLE 1,MSAに月/年を追加
safegraph_patterns = spark.sql("""
select x.*, INT(YEAR(FROM_UNIXTIME(date_range_start))) as year, 
                                        INT(MONTH(FROM_UNIXTIME(date_range_start))) as month, 
                                       case when region in ('NY', 'PA', 'NJ') then 'NYC MSA' else 'US' end msa, location_name
from safegraph_sim_foot_traffic x""")

# COMMAND ----------

# DBTITLE 1,データを拡張して日別の訪問者数を別の行に表示する
# Function to extract ARRAY or JSON columns for deeper analysis
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import *
from pyspark.sql.functions import *
import json

def parser(element):
  return json.loads(element)

def parser_maptype(element):
  return json.loads(element, MapType(StringType(), IntegerType()))

jsonudf = udf(parser, MapType(StringType(), IntegerType()))

convert_array_to_dict_udf = udf(lambda arr: {idx: x for idx, x in enumerate(json.loads(arr))}, MapType(StringType(), IntegerType()))

def explode_json_column_with_labels(df_parsed, column_to_explode, key_col="key", value_col="value"):
  df_exploded = df_parsed.select("safegraph_place_id", "location_name", "msa", "date_range_start", "year", "month", "date_range_end", explode(column_to_explode)).selectExpr("safegraph_place_id", "date_range_end", "location_name","msa", "date_range_start", "year", "month", "key as {0}".format(key_col), "value as {0}".format(value_col))
  return(df_exploded)

def explode_safegraph_json_column(df, column_to_explode, key_col="key", value_col="value"):
  df_parsed = df.withColumn("parsed_"+column_to_explode, jsonudf(column_to_explode))
  df_exploded = explode_json_column_with_labels(df_parsed, "parsed_"+column_to_explode, key_col=key_col, value_col=value_col)
  return(df_exploded)

def explode_safegraph_array_colum(df, column_to_explode, key_col="index", value_col="value"):
  df_prepped = df.select("safegraph_place_id", "location_name", "msa", "date_range_start", "year", "month", "date_range_end", column_to_explode).withColumn(column_to_explode+"_dict", convert_array_to_dict_udf(column_to_explode))
  df_exploded = explode_json_column_with_labels(df_prepped, column_to_explode=column_to_explode+"_dict", key_col=key_col, value_col=value_col)
  return(df_exploded)

def explode_safegraph_visits_by_day_column(df, column_to_explode, key_col="index", value_col="value"):
  df_exploded = explode_safegraph_array_colum(df, column_to_explode, key_col=key_col, value_col=value_col)
  df_exploded = df_exploded.withColumn(key_col, col(key_col) + 1) # 1-indexed instead of 0-indexed
  return(df_exploded)


visits_by_day = explode_safegraph_visits_by_day_column(safegraph_patterns, column_to_explode="visits_by_day", key_col="day", value_col="num_visits")
print(visits_by_day.count())
visits_by_day.createOrReplaceTempView('visits_by_day')

# COMMAND ----------

# MAGIC %sql select * from visits_by_day 

# COMMAND ----------

# DBTITLE 1,シルバテーブルをデルタ化(永続化)
visits_by_day.write.format('delta').mode('overwrite').save('/home/layla/data/table/footTrafficSilver')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Enrich the data with Subway Media data (Gold - Analytics Ready)
# MAGIC 
# MAGIC 
# MAGIC ##### Bring in various channels of media data: `banner impression`, `social media FB likes`, `web landing page visit`, `google trend` 

# COMMAND ----------

# DBTITLE 1,Subset the data to one particular Restaurant
# MAGIC %sql 
# MAGIC create
# MAGIC or replace temp view ft_raw as
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   visits_by_day;
# MAGIC create table if not exists layla.subway_foot_traffic as
# MAGIC select
# MAGIC   location_name,
# MAGIC   msa,
# MAGIC   year,
# MAGIC   month,
# MAGIC   day,
# MAGIC   num_visits
# MAGIC from
# MAGIC   ft_raw
# MAGIC where
# MAGIC   location_name = 'Subway';
# MAGIC select
# MAGIC   *
# MAGIC from
# MAGIC   layla_v2.subway_foot_traffic;

# COMMAND ----------

# DBTITLE 1,Load NYC Subway campaign media data; merge to foot traffic dataset
import numpy as np

city_pdf = spark.sql("select * from (select region, cast(year as integer) year, cast(month as integer) month, cast(day as integer) day, sum(num_visits) num_visits from layla_v2.Subway_foot_traffic  where  region = 'NY' and city = 'New York' group by region, cast(year as integer), cast(month as integer), cast(day as integer))").toPandas()

import pandas as pd
import numpy as np
city_pdf['date'] = pd.to_datetime(city_pdf[["year", "month", "day"]])
city_pdf = city_pdf.sort_values('date')
# generate NYC Subway campaign media: banner impression 
# normal distr
city_pdf['banner_imp'] = np.around( np.random.randint(20000, 100000, city_pdf.shape[0]) *  np.log(city_pdf['num_visits']))

#generate NYC Subway campaign media: social media like count
# lognormal distr
city_pdf['social_media_like'] = np.around( np.random.lognormal(3, 0.25, city_pdf.shape[0]) *  city_pdf['num_visits']/1000)

# generate landing page visit
# lognormal distr + moving average
city_pdf['landing_page_visit'] = np.around( np.random.lognormal(6, 0.03, city_pdf.shape[0]) * city_pdf['num_visits']/555).rolling(window=7).mean().fillna(400)

# COMMAND ----------

# MAGIC %md ##### Download [Google Trend data](https://trends.google.com/trends/explore?date=2019-01-01%202020-02-29&geo=US-NY-501&q=%2Fm%2F0f7q4) as index of organic search: can stream in by calling the Google Trend API using [`pytrends`](https://pypi.org/project/pytrends/)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/multiTimeline.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
google_trend_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(google_trend_df)

# Upsample from weekly to daily
google_trend_pdf = google_trend_df.toPandas().rename(columns={'Week': 'date'})
# google_trend_pdf['date'] = pd.to_datetime(google_trend_pdf['date'].str.strip(), format='%Y-%m-%d') ## need this line for DBR 7.1
merge=pd.merge(city_pdf,google_trend_pdf, how='left', on='date')
merge['google_trend_fill']  = merge['google_trend'].ffill()  
city_pdf = merge.fillna(82).drop('google_trend', axis=1).rename(columns={'google_trend_fill': 'google_trend'})

# COMMAND ----------

sdf = spark.createDataFrame(city_pdf)
sdf = sdf.withColumn("date", date_format(col("date"), "yyyy-MM-dd"))  #change to clean date format
display(sdf)

# COMMAND ----------

# DBTITLE 1,Daily TS chart of NYC Subway Foot Traffic from Jan-2019 to Feb-2020
display(sdf)

# COMMAND ----------

# DBTITLE 1,Write to Delta and create Gold Table
sdf.write.format('delta').mode('overwrite').save('/home/layla/data/table/footTrafficGold')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS layla.footTrafficGold 
# MAGIC USING DELTA
# MAGIC LOCATION '/home/layla/data/table/footTrafficGold'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS layla.footTrafficSilver 
# MAGIC USING DELTA
# MAGIC LOCATION '/home/layla/data/table/footTrafficSilver'
