# Databricks notebook source
## mount
dbutils.fs.mount(source='s3a://databricks-instacart', mount_point='/mnt/instacart')

## unmount
# dbutils.fs.unmount('/mnt/instacart')
