# Databricks notebook source
dbutils.notebook.run("CF 01: Data Preparation", 7200)

# COMMAND ----------

dbutils.notebook.run("CF 02: Identify Similar Users", 7200)

# COMMAND ----------

dbutils.notebook.run("CF 03: Build User-Based Recommendations", 7200)

# COMMAND ----------

dbutils.notebook.run("CF 04: Build Item-Based Recommendations", 7200)

# COMMAND ----------

dbutils.notebook.run("CF 05: Deploy Collaborative Filters", 7200)
