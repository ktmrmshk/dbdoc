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

# Deltaで保存する

delta_path = 'dbfs:/tmp/diamonds_cleaned.delta'
df_cleaned.write.format('delta').mode('overwrite').save(delta_path)

# COMMAND ----------

# MAGIC %md ## モデルのトレーニング

# COMMAND ----------

# 全部の特徴量を使って、単純な線形回帰を実施

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# loading data from delta 
df_cleaned = spark.read.format('delta').load(delta_path)

numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
cat_cols=['cut', 'color', 'clarity']

X = df_cleaned.select( numeric_cols + cat_cols ).toPandas()
y = df_cleaned.select('price').toPandas()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


preprocessor = ColumnTransformer([
  ('num', StandardScaler(), numeric_cols),
  ('cat', OneHotEncoder(), cat_cols)
])

lr = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('regressor', LinearRegression())
])

lr.fit(X_train, y_train)
lr.score(X_train, y_train)
pred = lr.predict(X_test)



# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression



preprocessor = ColumnTransformer([
  ('num', StandardScaler(), numeric_cols),
  ('cat', OneHotEncoder(), cat_cols)
])

lr = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('regressor', LinearRegression())
])


# loading data from delta 
df_cleaned = spark.read.format('delta').load(delta_path)

numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
cat_cols=['cut', 'color', 'clarity']

X = df_cleaned.select( numeric_cols + cat_cols ).toPandas()
y = df_cleaned.select('price').toPandas()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# train a model
lr.fit(X_train, y_train)
lr.score(X_train, y_train)
pred = lr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_test, pred, marker='.')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.show()

# COMMAND ----------

# MAGIC %md ## 特定の特徴量のみを使った線形回帰を関数化する

# COMMAND ----------

global_numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
global_cat_cols=['cut', 'color', 'clarity']

def do_train_model(selected_feature_cols, delta_path):
  '''
  selected_feature_cols: list of the features used for training
  '''
  
  # import libraries (for spark UDF env)
  from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import mlflow
  
  # Evaluate metrics
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2
  
  # split selected cols to numeric or categolized for preprocessing
  numeric_cols=[]
  cat_cols=[]
  for c in selected_feature_cols:
    if c in global_numeric_cols:
      numeric_cols.append(c)
    else:
      cat_cols.append(c)
      
  # make pipeline: packing preprocessing with model training
  preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(), cat_cols)
  ])

  lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
  ])  
  
  # load data from delta
  df_cleaned = spark.read.format('delta').load(delta_path)
  X = df_cleaned.select( numeric_cols + cat_cols ).toPandas()
  y = df_cleaned.select('price').toPandas()
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

  
  # fitting
  mlflow.sklearn.autolog()
  with mlflow.start_run():
    mlflow.log_param('selected_cols', )
    
    lr.fit(X_train, y_train)
    lr.score(X_train, y_train)
    pred = lr.predict(X_test)
  
    (rmse, mae, r2) = eval_metrics(test_y, pred)
    mlflow.log_param()
  
  
  
  

# COMMAND ----------

# MAGIC %md ## Sparkによる並列化

# COMMAND ----------

global_numeric_cols=['carat', 'depth', 'table', 'x', 'y', 'z']
global_cat_cols=['cut', 'color', 'clarity']

feature_cols = global_numeric_cols + global_cat_cols
print(feature_cols)

# COMMAND ----------

# 特徴量から4つのパラメータを選定するcombination
import itertools, json

feature_combs = list( itertools.combinations(feature_cols, 4) )
print(feature_combs)
df = spark.createDataFrame( feature_combs, ['feature_combination'])
display(df)

# COMMAND ----------

delta_path = 'dbfs:/tmp/diamonds_cleaned.delta'
df_cleaned.write.format('delta').mode('overwrite').save(delta_path)
display( df_cleaned )

# COMMAND ----------

from pyspark.sql.functions import col
display( df_cleaned.filter(  (col("x") < 3) | (col("price") < 500)  ) )

# COMMAND ----------


