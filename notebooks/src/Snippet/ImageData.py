# Databricks notebook source
# DBTITLE 1,サンプルのjpgファイル群
# MAGIC %fs ls /databricks-datasets/cctvVideos/train_images/label=0/

# COMMAND ----------

# MAGIC %md ###Imageのプレビュー(方法1) - DataFrame(`format="image"`) + `display()`
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC * SparkのDataframe(`format="image"`)で読み込む
# MAGIC * `display()`で表示する
# MAGIC 
# MAGIC ----
# MAGIC 
# MAGIC ドキュメント: [Image](https://docs.microsoft.com/ja-jp/azure/databricks/data/data-sources/image)

# COMMAND ----------

image_df = (
  spark
  .read
  .format("image")
  .load('/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame000*.jpg')
)

### DataFrameなので、いろいろ補助情報のカラムを追加できる
display(
  image_df
  .withColumn('filename', image_df.image.origin)
) 

# COMMAND ----------

# MAGIC %md ###Imageのプレビュー(方法2) - DataFrame(`format="binary"`) + `display()`
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC * SparkのDataframe(`format="binaryFile"`)で読み込む
# MAGIC * `.option("mimeType", "image/*")`でコンテンツ種別を指定
# MAGIC * `display()`で表示する
# MAGIC 
# MAGIC ---
# MAGIC 
# MAGIC ドキュメント: [Binary File](https://docs.databricks.com/data/data-sources/binary-file.html)

# COMMAND ----------

image_df = (
  spark
  .read
  .format("binaryFile")
  .option("mimeType", "image/*")
  .load('/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame000*.jpg')
)

display( image_df ) 

# COMMAND ----------

# MAGIC %md ###Imageのプレビュー(方法3) - PIL + numpy + matplotlib
# MAGIC ----
# MAGIC 
# MAGIC * 通常のJupyter Notebook上で実施するのと同じ方法

# COMMAND ----------

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

pil_im = Image.open('/dbfs/databricks-datasets/cctvVideos/train_images/label=0/Browse2frame0002.jpg', 'r')
imshow(np.asarray(pil_im))
