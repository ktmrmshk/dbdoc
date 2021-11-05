# Databricks notebook source
df = spark.read.format('csv').option('Header', True).load('/FileStore/ktmr-test/2020_12_2021_11.csv')
display(df.filter(df.sku == 'ENTERPRISE_ALL_PURPOSE_COMPUTE'))
df.createOrReplaceTempView('usage')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM usage
# MAGIC where clusterName = "cluster_for_handson_renew"

# COMMAND ----------

# MAGIC %fs ls /mnt/instacart/

# COMMAND ----------

# MAGIC %fs ls s3a://databricks-instacart/

# COMMAND ----------

dbutils.fs.mount(source='s3a://databricks-instacart', mount_point='/mnt/instacart')

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/cctvVideos/train_images/label=0/

# COMMAND ----------

# from IPython.display import Image 
# pil_img = Image(filename='/dbfs/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame0002.jpg')
# IPython.display(pil_img)

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

#%matplotlib inline
pil_im = Image.open('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame0002.jpg', 'r')
imshow(np.asarray(pil_im))

# COMMAND ----------

from PIL import Image
pil_im = Image.open('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame0002.jpg', 'r')
pil_im.show()

# COMMAND ----------

image_df = spark.read.format("image").load('/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame000*.jpg')

display(image_df.withColumn('filename', image_df.image.origin)) 

# COMMAND ----------

df = spark.read.format('csv').load(
  '/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv',
  header=True,
  inferSchema=True
)
dbutils.data.summarize(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT count(1) FROM ktmr_cloudtrail_jobs.cloud_trail_bronze

# COMMAND ----------

# MAGIC %fs ls /FileStore/ktmr/spark_examples_2_12_3_1_2.jar

# COMMAND ----------

df = spark.read.format('csv').option('Header', True).load('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv.zip')
display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists ktmrdb.sandtable
# MAGIC ( id int, name string, age int, description string )
# MAGIC using delta
# MAGIC location 's3://databricks-ktmr-s3/sandtable.delta'

# COMMAND ----------

# MAGIC %sql 
# MAGIC insert into ktmrdb.sandtable (id, name) values ( 123, 'suzuki')

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from 
# MAGIC ktmr_cloudtrail_jobs.cloud_trail_bronze

# COMMAND ----------

41844696 - 84984935

# COMMAND ----------

# MAGIC %fs ls /FileStore/ktmr/foo.txt

# COMMAND ----------

spark.range(100).show()

# COMMAND ----------

sc.defaultParallelism

# COMMAND ----------

spark.range(100).crossJoin( spark.range(2,21).withColumnRenamed('id', 'n')).repartition(sc.defaultParallelism).select('n').rdd.take(10) 


# COMMAND ----------

df = spark.read.format('csv').option('Header', True).load('s3://databricks-ktmr-s3/sample/airbnb/AB_NYC_2019.csv')

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

df2.write.format('parquet').mode('overwrite').save('s3://databricks-ktmr-s3/tmp/foo.parquet')

# COMMAND ----------

df2= df.repartition(1)

# COMMAND ----------

df.write.parquet("s3://databricks-ktmr-s3/tmp/foo.parquet")

# COMMAND ----------

# MAGIC %fs ls /mnt/

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists ktmrdb.test123
# MAGIC ( name string, age int)
# MAGIC using delta 
# MAGIC location 's3://databricks-ktmr-s3/foobar.delta'

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into ktmrdb.test123
# MAGIC values ('kitamura', 123)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from ktmrdb.test123

# COMMAND ----------

df = spark.read.format('delta').load('s3://databricks-ktmr-s3/foobar.delta')
display( df.withColumnRenamed('name', 'なまえ') )

spark
.read.format('delta').load('s3://バケツ名/foobar.delta')
.withColumnRenamed('name', 'なまえ')
.write.format('delta')
.mode('overwrite')
.save('s3://バケツ名/foobar.delta')

# COMMAND ----------

# MAGIC %sql
# MAGIC describe ktmrdb.test123

# COMMAND ----------

# MAGIC %sql
# MAGIC alter table ktmrdb.test123 rename column name to namae;

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table ktmrdb.test123;

# COMMAND ----------

# MAGIC %fs rm -r s3://databricks-ktmr-s3/foobar.delta

# COMMAND ----------

# MAGIC %sql
# MAGIC create table if not exists ktmrdb.test123
# MAGIC ( `あああ` int, name string, age int)
# MAGIC using delta 
# MAGIC location 's3://databricks-ktmr-s3/foobar.delta'

# COMMAND ----------


