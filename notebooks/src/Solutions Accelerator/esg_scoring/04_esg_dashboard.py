# Databricks notebook source
# MAGIC %md
# MAGIC # ESG - dashboard
# MAGIC 
# MAGIC The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their **environmental, social and governance** (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). In this demo, we offer a novel approach to sustainable finance by combining NLP techniques and graph analytics to extract key strategic ESG initiatives and learn companies' relationships in a global market.
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
# MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
# MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
# MAGIC + <a href="$./04_esg_dashboard">STAGE4</a>: Package all visualizations into powerful dashboards
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC This notebook summarizes what we learned through NLP and graph analytics with visualisations in order to provide asset managers and Chief Risk Officers with a 360 view on their ethical investments strategies. Using the built-in dashboard functionality (see the [associated dashboard](https://demo.cloud.databricks.com/#notebook/7135966/dashboard/7136926/present)) or the agility providing by [redash](https://redash-demo.dev.databricks.com/public/dashboards/rHGDzBNRvvomCTyKHRjB4STqha77VDQAM0HzfgZX?org_slug=default), this series of insights can be rapidly packaged up as a BI/MI report, bringing ESG as-a-service to your organisation for asset managers to better invest in sustainable and responsible finance.
# MAGIC 
# MAGIC ### Dependencies
# MAGIC 
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. **NOTE** The next cell assumes you are running this notebook on a Databricks cluster that does not make use of the ML runtime.  If using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. 

# COMMAND ----------

# MAGIC %pip install plotly wordcloud

# COMMAND ----------

# DBTITLE 1,Install needed libraries
# dbutils.library.installPyPI('wordcloud')
# dbutils.library.installPyPI('plotly')
# dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import libraries
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf

from sklearn.feature_extraction import text 
from sklearn.preprocessing import MinMaxScaler

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import random

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP1`: prepare notebook with data views
# MAGIC These will be reference data common across all selected organisation that will be loaded only once and cached / collected to memory

# COMMAND ----------

# DBTITLE 1,Find country of operation
# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW esg_gdelt_countries AS
# MAGIC SELECT 
# MAGIC   u.organisation,
# MAGIC   u.country,
# MAGIC   SUM(u.tone) AS tone,
# MAGIC   COUNT(*) AS total
# MAGIC FROM (
# MAGIC   SELECT 
# MAGIC     to_date(g.publishDate) AS date, -- convert timestamp to date
# MAGIC     g.organisation,
# MAGIC     explode(g.countries) AS country,
# MAGIC     g.tone
# MAGIC   FROM esg.gdelt_silver g
# MAGIC ) u
# MAGIC WHERE length(u.country) = 3
# MAGIC GROUP BY
# MAGIC   u.organisation,
# MAGIC   u.country;
# MAGIC   
# MAGIC CACHE TABLE esg_gdelt_countries;

# COMMAND ----------

# DBTITLE 1,List of organisations to report in widget
organisations = [
  "standard chartered",
  "rbc",
  "credit suisse",
  "goldman sachs",
  "jp morgan chase",
  "lazard",
  "macquarie",
  "barclays",
  "northern trust",
  "citi",
  "morgan stanley"
]

# COMMAND ----------

# DBTITLE 1,Extract topic distribution across industries
topics_total = spark \
  .read \
  .table('esg.reports') \
  .groupBy('topic', 'company') \
  .count() \
  .groupBy('topic') \
  .agg(F.avg('count').alias('average')) \
  .withColumn('organisation', F.lit('industry')) \
  .select('topic', 'organisation', 'average') \
  .toPandas() \
  .sort_values('topic')

topics_total.index = topics_total.topic
topics_total = topics_total.drop(['organisation', 'topic'], axis=1)

# COMMAND ----------

# DBTITLE 1,Create aggregated themes view
theme_sentiment_industry = spark \
  .read \
  .table("esg.gdelt_gold") \
  .filter(F.col("organisation").isin(organisations)) \
  .groupBy("date", "theme") \
  .agg(
    F.sum("tone").alias("tone"),
    F.sum("total").alias("total")
  ) \
  .withColumn("industry_average", F.col("tone") / F.col("total")) \
  .withColumn("industry_total", F.col("total")) \
  .select("date", "theme", "industry_average", "industry_total") \
  .orderBy(F.asc("date"))

theme_sentiment_industry.cache()
theme_sentiment_industry.count()

# COMMAND ----------

# DBTITLE 1,Create aggregated URLs view
window = Window.partitionBy("organisation").orderBy(F.asc("tone"))

esg_articles = spark \
  .read \
  .table("esg.gdelt_silver") \
  .filter(F.col("organisation").isin(organisations)) \
  .withColumn("theme", F.explode(F.col("themes"))) \
  .filter(F.col("theme") == 'E') \
  .withColumn('rank', F.dense_rank().over(window)) \
  .filter(F.col('rank') < 100) \
  .select(F.col("organisation"), F.to_date(F.col("publishDate")).alias("date"), F.col("url"), F.col("tone"))

esg_articles.cache()
esg_articles.count()

# COMMAND ----------

# DBTITLE 1,Normalize propagated weighted ESG scores
# read all internal ESG scores
gdelt_esg_all = spark \
    .read \
    .table("esg.scores") \
    .select("organisation", "theme", "esg")

# read internal ESG scores for our FSIs
gdelt_esg_org = spark \
    .read \
    .table("esg.scores") \
    .filter(F.col("organisation").isin(organisations)) \
    .select("organisation", "theme", "esg")

# read propagated weighted ESG scores for our FSIs
gdelt_esg_org_pr = spark \
    .read \
    .table("esg.scores_fsi") \
    .filter(F.col("organisation").isin(organisations)) \
    .select("organisation", "theme", "esg")

esg_scores = {}
for theme in ['E', 'S', 'G']:
  
  # convert spark to Pandas DF
  gdelt_theme_all = gdelt_esg_all[gdelt_esg_all['theme'] == theme].drop("theme").toPandas().set_index("organisation")
  gdelt_theme_org = gdelt_esg_org[gdelt_esg_org['theme'] == theme].drop("theme").toPandas().set_index("organisation")
  gdelt_theme_org_pr = gdelt_esg_org_pr[gdelt_esg_org_pr['theme'] == theme].drop("theme").toPandas().set_index("organisation")
  
  # retrieve max and min
  esg_min = gdelt_theme_all.min()
  esg_max = gdelt_theme_all.max()

  # normalize scores between 0 and 100 
  gdelt_theme_org = 100 * (gdelt_theme_org - esg_min)/(esg_max - esg_min)
  gdelt_theme_org_pr = 100 * (gdelt_theme_org_pr - esg_min)/(esg_max - esg_min)
  gdelt_theme_org_pr = gdelt_theme_org_pr.rename(columns={'esg': 'esg_pr'})
  
  # merge both internal and weighted propagated ESGs
  esg_scores[theme] = gdelt_theme_org_pr.merge(gdelt_theme_org, left_index=True, right_index=True)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP2`: generic visualisations
# MAGIC These will be generic visualisations such as the comparaison of ESG topics for each company

# COMMAND ----------

# DBTITLE 1,Industry comparaison
# create a simple pivot table of number of occurence of each topic across organisations
import uuid

esg_group = spark.read.table('esg.reports').filter(F.col("probability") > 0.9).toPandas()
esg_focus = pd.crosstab(esg_group.company, esg_group.topic)

# scale topic frequency between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))

# normalize pivot table
esg_focus_norm = pd.DataFrame(scaler.fit_transform(esg_focus), columns=esg_focus.columns)
esg_focus_norm.index = esg_focus.index

# plot heatmap, showing main area of focus for each FSI across topics we learned
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(esg_focus_norm, annot=False, linewidths=.5, cmap='Blues')
plt.xlabel('')
plt.ylabel('')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # `STEP3`: dynamic visualisations
# MAGIC These visualisations will be dynamically updated when selecting organisation using provided widgets

# COMMAND ----------

# DBTITLE 1,Create interactive widget
try:
  dbutils.widgets.remove('organisation')
except:
  pass

dbutils.widgets.dropdown('organisation', organisations[0], organisations)

# COMMAND ----------

# DBTITLE 1,Get top keywords on ESG report
# context specific keywords not to include in topic modelling
fsi_stop_words = [
  '2018', '2019', 'jan', 'dec',
  'plc', 'group', 'target',
  'track', 'capital', 'holding',
  'report', 'annualreport',
  'esg', 'bank', 'report',
  'annualreport', 'long', 'make'
]

# color wordcloud with blue palette
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(188, 94%%, %d%%)" % random.randint(25, 45)

# add company names as stop words
org = dbutils.widgets.get('organisation')
for t in org.split(' '):
  fsi_stop_words.append(t)

# our list contains all english stop words + companies names + specific keywords
stop_words = text.ENGLISH_STOP_WORDS.union(fsi_stop_words)

# retrieve all ESG statements for the given organisation
lemma_df = spark \
  .read \
  .table('esg.reports') \
  .filter(F.col('company') == org) \
  .toPandas().lemma

# we simply aggregate all sentences to use wordcloud
long_string = ' '.join(list(lemma_df))

# define our wordcloud object
wc = WordCloud(
    background_color="white",
    max_words=5000, 
    width=600, 
    height=400, 
    stopwords=stop_words, 
    contour_width=3, 
    contour_color='steelblue'
)

# train our wordcloud
wc.generate(long_string)
  
plt.figure(figsize=(8,8))
plt.imshow(wc.recolor(color_func=color_func, random_state=3), interpolation='bilinear')
plt.axis("off")
plt.show()

# COMMAND ----------

# DBTITLE 1,Get key initiatives
# we do not simply extract each topic, but rank topics by the distance to closest kmeans cluster
# this will return statements that are different from all competitors, therefore specific to that organisation
# "everybody wants to save Antartica, but concretely, what is your plan?"
window = Window.partitionBy("topic").orderBy(F.desc("probability"))

# we apply the 90% threshold observed in previous notebook to extract topic specific statements
# rank these statements using partitioning window
initiatives = spark \
  .read \
  .table('esg.reports') \
  .filter(F.col('company') == dbutils.widgets.get('organisation')) \
  .filter(F.col('probability') > 0.9) \
  .withColumn('statement', F.first(F.col('statement')).over(window)) \
  .select('topic', 'statement') \
  .distinct()

display(initiatives)

# COMMAND ----------

# DBTITLE 1,Create Gauge
import plotly.graph_objects as go

def gauge(esg_norm, esg, title):
  """
  Create a gauge using plotly, indicating score and the difference with internal ESG score (without page rank)
  """
  fig = go.Figure(go.Indicator(
      domain = {'x': [0, 1], 'y': [0, 1]},
      value = esg_norm,
      mode = "gauge+number+delta",
      delta = {'reference': esg},
      gauge = {'bar': {'color': "lightblue"}, 'axis': {'range': [None, 100]}}))

  fig.show()

# COMMAND ----------

# DBTITLE 1,Environment
# retrieve ESG score for environment themes
org = dbutils.widgets.get('organisation')
try:
  scores = esg_scores['E'].loc[org]
  gauge(scores.esg_pr, scores.esg, 'E')
except:
  pass

# COMMAND ----------

# DBTITLE 1,Social
# retrieve ESG score for environment social
org = dbutils.widgets.get('organisation')
try:
  scores = esg_scores['S'].loc[org]
  gauge(scores.esg_pr, scores.esg, 'S')
except:
  pass

# COMMAND ----------

# DBTITLE 1,Governance
# retrieve ESG score for environment governance
org = dbutils.widgets.get('organisation')
try:
  scores = esg_scores['G'].loc[org]
  gauge(scores.esg_pr, scores.esg, 'G')
except:
  pass

# COMMAND ----------

# DBTITLE 1,Topic distributions
org = dbutils.widgets.get('organisation')

# extract topics for a specific org
topics = spark \
  .read \
  .table('esg.reports') \
  .filter(F.col('company') == org) \
  .withColumnRenamed('company', 'organisation') \
  .groupBy('topic', 'organisation') \
  .count() \
  .withColumnRenamed('count', 'total') \
  .select('topic', 'organisation', 'total') \
  .toPandas() \
  .sort_values('topic')

topics.index = topics.topic
topics = topics.drop(['organisation', 'topic'], axis=1)

# merge with industry topics
df = topics.merge(topics_total, left_index=True, right_index=True)
labels = list(df.index)
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(20,7))
rects1 = ax.bar(x - width/2, df.total, width, color='steelblue', alpha=0.8, label=org)
rects2 = ax.bar(x + width/2, df.average, width, color='lightblue', alpha=0.4, label='industry average')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_title('')
ax.set_xticks(x)
ax.set_yticks([])
ax.set_xticklabels(labels, fontsize=10)
ax.xaxis.set_tick_params(rotation=0)
ax.legend(frameon=False)

# COMMAND ----------

# DBTITLE 1,Retrieve sentiment time series
def sentiment_series(org, theme):
  
  # read dataframe of sentiment over time
  theme_sentiment = spark \
      .read \
      .table('esg.gdelt_gold') \
      .filter(F.col('organisation') == org) \
      .filter(F.col('theme') == theme) \
      .withColumn("tone", F.col("tone") / F.col("total")) \
      .join(theme_sentiment_industry, ['date', 'theme']) \
      .select('date', 'tone', 'industry_average') \
      .toPandas()
  
  # complete time series with missing values (last observation carried forward)
  theme_sentiment = theme_sentiment.sort_values('date')
  theme_sentiment = theme_sentiment.set_index('date')
  theme_sentiment = theme_sentiment.asfreq(freq = 'D', method = 'pad')
  
  # apply a 30 days moving average to smooth signal
  theme_sentiment = theme_sentiment.rolling(window=30).mean()
  theme_sentiment['delta'] = theme_sentiment['tone'] - theme_sentiment['industry_average']
  theme_sentiment['date'] = theme_sentiment.index

  return theme_sentiment

# COMMAND ----------

# DBTITLE 1,ESG scores
themes = ['E', 'S', 'G']
fig, axs = plt.subplots(3)
org = dbutils.widgets.get('organisation')

# retrieve sentiment for each E, S, G and compare with industry
# we simply look at delta to see how a company is positioned
for i, theme in enumerate(themes):
  ax = axs[i]
  df = sentiment_series(org, theme).dropna()[['delta']]
  ax.plot(df.index, df.delta, label=org + ' ({})'.format(theme), linewidth=2, color='dodgerblue')
  ax.xaxis_date()
  ax.axhline(0, linestyle='--', linewidth=.8, color='coral')
  ax.grid(False)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.legend(loc='upper left', frameon=False)

plt.show()

# COMMAND ----------

# DBTITLE 1,ESG sentiment analysis
org = dbutils.widgets.get('organisation')

# read dataframe of sentiment over time
sentiment = spark \
  .read \
  .table('esg.gdelt_gold') \
  .filter(F.col('organisation') == org) \
  .groupBy('date') \
  .agg(
    F.sum("tone").alias("tone"),
    F.sum("total").alias("total")
  ) \
  .withColumn("tone", F.col("tone") / F.col("total")) \
  .toPandas()[['date', 'tone', 'total']]

# complete time series with missing values (last observation carried forward)
sentiment = sentiment.sort_values('date')
sentiment = sentiment.set_index('date')
sentiment = sentiment.asfreq(freq = 'D', method = 'pad')

plt.figure(figsize=(20,6))
plt.plot(sentiment.index, sentiment.tone, color='lightblue', label='')
# apply a 14 days moving average to smooth signal
plt.plot(sentiment.index, sentiment.rolling(window=14).mean().tone, color='dodgerblue', linewidth=2, label='sentiment')
plt.axhline(0, linewidth=.5, color='grey')
plt.legend(loc='upper left', frameon=False)
plt.show()

# COMMAND ----------

# DBTITLE 1,Impact on world
# display all the countries mentioned alongside organisation, color coding countries with positive sentiment
# this shows where companies had a positive impact to
display(
  spark \
    .read \
    .table("esg_gdelt_countries") \
    .filter(F.col('organisation') == dbutils.widgets.get('organisation')) \
    .filter(F.length(F.col("country")) == 3) \
    .withColumn("tone", F.col("tone") / F.col("total"))
)

# COMMAND ----------

# DBTITLE 1,Access news articles
# retrieve all news articles for a specific organisation, worst articles (in term of sentiment first)
display(
  esg_articles
    .filter(F.col('organisation') == dbutils.widgets.get('organisation')) \
    .orderBy(F.asc("tone")).drop("organisation") \
    .limit(200)
)

# COMMAND ----------

# DBTITLE 1,Show connections
org = dbutils.widgets.get('organisation')

# retrieve companies mentioned alongside organisation
# we use the importance we've learned through personalised page rank to build a wordcloud
connection_df = spark \
    .read \
    .table("esg.connections") \
    .filter(F.col('organisation') == dbutils.widgets.get('organisation')) \
    .filter(F.col('connection') != dbutils.widgets.get('organisation')) \
    .toPandas()[['connection', 'importance']]

connections = {}
for i, row in connection_df.iterrows():
  connections[row.connection] = row.importance

# blue palette
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(188, 94%%, %d%%)" % random.randint(25, 45)

if (len(connections) > 0):
  wc = WordCloud(
      background_color="white",
      max_words=5000, 
      width=600, 
      height=400, 
      contour_width=3, 
      contour_color='steelblue'
  )

  plt.figure(figsize=(8,8))
  wc.generate_from_frequencies(connections)
  plt.imshow(wc.recolor(color_func=color_func, random_state=3), interpolation='bilinear')
  plt.axis("off")
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
# MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
# MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
# MAGIC + <a href="$./04_esg_dashboard">STAGE4</a>: Package all visualizations into powerful dashboards
# MAGIC ---

# COMMAND ----------


