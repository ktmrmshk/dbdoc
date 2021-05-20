# Databricks notebook source
# MAGIC %md
# MAGIC # SecretKey, AccessKeyを使ってS3にアクセスする
# MAGIC 
# MAGIC AWSのSecretKey, AccessKeyを使ってDatabricksからアクセスするサンプルになります。
# MAGIC 
# MAGIC **注意**
# MAGIC 
# MAGIC コードは簡単のため、Secret Key, Access Keyが平文で書かれています。そのため、本番環境では使用する場合は、Databricksの[Secret機能](https://docs.databricks.com/security/secrets/index.html)を用いて安全にSecret Key, Access Key管理する方法や、[Instance ProfileによるIAM Roleベースのアクセス](https://docs.databricks.com/administration-guide/cloud-configurations/aws/instance-profiles.html)方法を検討ください。

# COMMAND ----------

# AWSのAccessKey, SecretKeyを指定
access_key = 'your_access_key'
secret_key = 'your_secret_key'
encoded_secret_key = secret_key.replace("/", "%2F")

# S3のバケツを指定
s3bucket = 'bucket_name'



# パスを構成する
file_path = f's3a://{access_key}:{encoded_secret_key}@{s3bucket}/'
print('file_path => ', file_path)

### アクセス方法1. そのままパスを指定して参照する方法 => secret/access keyがそのまま表示される
display( dbutils.fs.ls(file_path) )


### アクセス方法2. DBFS(Databricksが使用するファイルシステムのツリー)にmountしてから使用する
mount_point = '/mnt/FOO_BAR_POINT'
try:
  dbutils.fs.mount(file_path, mount_point)
except Exception as e:
  print('already mounted: ', e)

display( dbutils.fs.ls(mount_point) )

# unmountする
dbutils.fs.unmount(mount_point)

# COMMAND ----------


