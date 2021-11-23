# Databricks notebook source
# MAGIC %md ## データの準備・クレンジング・Delta化

# COMMAND ----------

diamonds_schema='''
id int,
carat double,
cut string,
color string,
clarity string,
depth double,
table int,
price int,
x double,
y double,
z double
'''

df = spark.read.format('csv').option('Header', True).schema(diamonds_schema).load('/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv')
display(df)
df.printSchema()

# COMMAND ----------

dbutils.data.summarize(df)

# COMMAND ----------

# nullのレコード数を確認する
print(df.count())
print(df.filter('table is null').count() )

# COMMAND ----------

# 今回はnullレコードが小さいので、トレーニングデータから除外する方針で進める
df_cleaned = df.where('table is not null')

# COMMAND ----------

# 再度、nullがないかを確認
dbutils.data.summarize( df_cleaned)

# COMMAND ----------

# Deltaで保存する(永続化する)

delta_path = 'dbfs:/tmp/diamonds_cleaned.delta'
df_cleaned.write.format('delta').mode('overwrite').save(delta_path)

# COMMAND ----------

# MAGIC %md ## モデルのトレーニング

# COMMAND ----------

# DBTITLE 1,Regressor(アルゴリズム)を引数に学習部分を関数化
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


def do_train_model(regressor):

  # 変数設定
  delta_path = '/tmp/diamonds_cleaned.delta'
  numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
  cat_cols=['cut', 'color', 'clarity']
  
  preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), cat_cols)
  ])

  pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
  ])


  # loading data from delta 
  df_cleaned = spark.read.format('delta').load(delta_path)

  numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
  cat_cols=['cut', 'color', 'clarity']

  X = df_cleaned.select( numeric_cols + cat_cols ).toPandas()
  y = df_cleaned.select('price').toPandas()
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


  # train a model
  pipeline.fit(X_train, y_train)
  print('test:', pipeline.score(X_train, y_train) )

  pred = pipeline.predict(X_test)
  print('validation:', pipeline.score(X_test, y_test) )


  # 3. プロット
  import matplotlib.pyplot as plt
  plt.scatter(y_test, pred, marker='.')
  plt.xlabel('actual')
  plt.ylabel('predicted')
  plt.show()
  
  return pipeline

# COMMAND ----------

# DBTITLE 1,線形回帰
print('=== LinearRegression ===')
lr = LinearRegression()
do_train_model(lr)


# COMMAND ----------

# DBTITLE 1,ランダムフォレスト
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=10, random_state=42, n_estimators=100)
pl = do_train_model(rf)



# COMMAND ----------

pl.named_steps['regressor'].feature_importances_

# COMMAND ----------

# DBTITLE 1,Lasso回帰(制限強め)
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=50)
pl = do_train_model(lasso)

# COMMAND ----------

pl.named_steps['regressor'].coef_

# COMMAND ----------

# DBTITLE 1,Gradient Boosting Tree
from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(random_state=0, max_depth=10)
pl = do_train_model(reg)


# COMMAND ----------

pl.named_steps['regressor'].feature_importances_

# COMMAND ----------

# MAGIC %md ## `cut`特徴量ごとにモデルを作る(Sparkで並列分散化)

# COMMAND ----------

# DBTITLE 1,再度、データをロードしておく
df_cleaned = spark.read.format('delta').load('/tmp/diamonds_cleaned.delta')
display(df_cleaned)
print( df_cleaned.count())

# COMMAND ----------

# DBTITLE 1,`cut`ごとのレコード数
display( 
  df_cleaned.groupBy('cut').count()
)

# COMMAND ----------

# DBTITLE 1,`cut`ごとに分解されたPandas dataframe(p_df)を受け取って学習する部分を関数化
# 学習するモデルはMLflowでトラックする
# モデルの性能比較はMLflow上で行う
# のちにモデルが必要になったらMLflowからロードする

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mlflow
import json
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt


def do_train_model(regressor, p_df):
  
  # 変数設定
  numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
  cat_cols=['color', 'clarity']
  
  preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), cat_cols)
  ])

  pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
  ])

  numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
  cat_cols=['cut', 'color', 'clarity']

  X = p_df[ numeric_cols + cat_cols ]
  y = p_df['price']
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

  # train a model
  score=0
  mlflow.sklearn.autolog()
  with mlflow.start_run() as run:
    mlflow.log_param('regressor', str(regressor))
    mlflow.log_param('cut', p_df['cut'].iloc[0])
    
    pipeline.fit(X_train, y_train)
    mlflow.log_metric('train_score', pipeline.score(X_train, y_train) )
    
    pred = pipeline.predict(X_test)
    score = pipeline.score(X_test, y_test)
    mlflow.log_metric('validation_score', score )
    
    # Plot
    plt.scatter(y_test, pred, marker='.')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.savefig('scoreplot.png')
    mlflow.log_artifact('scoreplot.png')
  
  # 後処理
  p_df['score'] = score
  p_df['runid'] = run.info.run_id
  return p_df[ ['cut', 'score', 'runid']  ]




# COMMAND ----------

# DBTITLE 1,線形回帰 - LinearRegression()
from sklearn.linear_model import LinearRegression

@pandas_udf('cut string, score double, runid string', PandasUDFType.GROUPED_MAP)
def lr_do_train_model(p_df):
  lr = LinearRegression()
  return do_train_model(lr, p_df)

df_ret = df_cleaned.groupBy('cut').apply(lr_do_train_model)
display(df_ret.distinct())

# COMMAND ----------

# DBTITLE 1,ランダムフォレスト - RandomForestRegressor(()
from sklearn.ensemble import RandomForestRegressor

@pandas_udf('cut string, score double, runid string', PandasUDFType.GROUPED_MAP)
def rf_do_train_model(p_df):
  rf = RandomForestRegressor(max_depth=10, random_state=42, n_estimators=100)
  return do_train_model(rf, p_df)

df_ret = df_cleaned.groupBy('cut').apply(rf_do_train_model)
display(df_ret.distinct())


# COMMAND ----------

# DBTITLE 1,Gradient Boosting - GradientBoostingRegressor()
from sklearn.ensemble import GradientBoostingRegressor

@pandas_udf('cut string, score double, runid string', PandasUDFType.GROUPED_MAP)
def gb_do_train_model(p_df):
  gb = GradientBoostingRegressor(random_state=0, max_depth=10)
  return do_train_model(gb, p_df)

df_ret = df_cleaned.groupBy('cut').apply(gb_do_train_model)
display(df_ret.distinct())

# COMMAND ----------

# DBTITLE 1,MLflowからモデルをロードする
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.download_artifacts('36fe55737d974d14bc7b9faf3b3192ee', 'model/model.pkl', '/tmp/')

import pickle
model=None
with open('/tmp/model/model.pkl', 'rb') as f:
  model=pickle.load(f)

regressor = model.named_steps['regressor']
print(regressor)
print(regressor.feature_importances_)

# COMMAND ----------


