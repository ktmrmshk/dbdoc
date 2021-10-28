# Databricks notebook source
# MAGIC %md This notebook was developed for use with a **Databricks 8.4 ML** cluster. It provides access to consistent configuration settings across the notebooks that make up this solution accelerator.  It also provides instructions for the setup of the files required by these notebooks.

# COMMAND ----------

# DBTITLE 1,Initialize Config Settings
if 'config' not in locals():
  config = {}

# COMMAND ----------

# MAGIC %md The basic building block of our market basket recommender is transactional data identifying the products customers purchased together at checkout. The popular [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis) provides us a nice collection of such data with over 3.3 million grocery orders placed by over 200,000 Instacart users over a nearly 2-year period across of portfolio of nearly 50,000 products.
# MAGIC 
# MAGIC **NOTE** Due to the terms and conditions by which these data are made available, anyone interested in recreating this work will need to download the data files from Kaggle and upload them to a folder structure as described below.
# MAGIC 
# MAGIC The primary data files from this dataset should be uploaded to cloud storage and organized as follows under a pre-defined [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) that we have named */mnt/instacart*:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=250>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC The path of the mount point and the folders containing these files are specified as follows:

# COMMAND ----------

# DBTITLE 1,File Path Configurations
config['mount_point'] = '/mnt/instacart'
config['orders_files'] = config['mount_point'] + '/bronze/orders'
config['products_files'] = config['mount_point'] + '/bronze/products'
config['order_products_files'] = config['mount_point'] + '/bronze/order_products'
config['departments_files'] = config['mount_point'] + '/bronze/departments'
config['aisles_files'] = config['mount_point'] + '/bronze/aisles'

# COMMAND ----------

# MAGIC %md © 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
