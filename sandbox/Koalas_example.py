# Databricks notebook source
import pandas as pd
import numpy as np
import databricks.koalas as ks

# COMMAND ----------

k_df = ks.read_csv('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv', header=0)
k_df.head()

# COMMAND ----------

k_s = ks.Series([1,2,3,4, np.nan])
k_s

# COMMAND ----------

k_df = ks.DataFrame({'a':[1,2,3,4], 'b': ['a', 'b', 'c', '']}, index = [10,20,30,40])
k_df

# COMMAND ----------

k_df.describe()

# COMMAND ----------

k_df.columns

# COMMAND ----------

k_df.index

# COMMAND ----------

k_df.mean()

# COMMAND ----------

k_df.T

# COMMAND ----------


