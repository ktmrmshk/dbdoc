# Databricks notebook source
# MAGIC %md # DataFrame内の画像ファイル(JPG)を一括で画像処理し、その後Exportする

# COMMAND ----------

# MAGIC %md ### 1. サンプルのImage Dataframeを作成

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

# MAGIC %md ### 2. 画像処理(グレイスケール)を一括でかける

# COMMAND ----------

# MAGIC %md #### 2.1  画像処理をUDFにする

# COMMAND ----------

@udf('binary')
def convert_grayscale(content):
  '''
  contentをグレイスケール変換する
  
  param: 
    content: Imageのバイナリデータのカラム
  return: 
    procced_image_binary: 画像処理後のバイナリ 
  '''
  import os, io
  from PIL import Image
  
  # `content`カラム = imageバイナリの読み込み
  f=io.BytesIO(content)
  im = Image.open(f)
  
  # 画像処理
  grayscaled_im = im.convert('L')
  
  # imageバイナリを返却する
  out = io.BytesIO()
  grayscaled_im.save(out, format='JPEG')
  return out.getvalue()

# COMMAND ----------

# MAGIC %md ### 2.2 UDFを適用して、画像処理(grayscale変換)を実施

# COMMAND ----------

grayscaled_df = image_df.withColumn('grayscaled_content', convert_grayscale('content'))

display( 
  grayscaled_df
)

# COMMAND ----------

# MAGIC %md #### 2.3 変換後の画像を一枚取り出して、プレビュー確認

# COMMAND ----------

import io
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

byte_image = grayscaled_df.limit(1).select('grayscaled_content').collect()[0]['grayscaled_content'] # <=変換後(グレイスケール)
#byte_image = grayscaled_df.limit(1).select('content').collect()[0]['content'] #<= オリジナル

f=io.BytesIO(byte_image)
im = Image.open(f, formats=['JPEG'])
imshow(np.asarray(im), cmap = "gray")


# COMMAND ----------

# MAGIC %md ### 3. Dataframe内の画像ファイルの一括でExportする

# COMMAND ----------

# MAGIC %md #### 3.1  画像ファイルをExportするするためのUDFを作成

# COMMAND ----------

@udf('string')
def export_as_jpg(path, content):
  '''
  contentをJPGファイルとして出力する。
  
  param: 
    path: ファイルパスのカラム
    content: Imageのバイナリデータのカラム
  return: 
    export_path: 出力先のパス
  
  '''
  import os, io
  from PIL import Image
  
  # ファイルの出力ファイル名を構成
  # オリジナルが`image001.jpg`であれば、`image001_proc.jgp`として出力
  base=os.path.basename(path)
  basename, ext = os.path.splitext(base)
  export_filename=f'{basename}_proc{ext}'
  
  export_base='/dbfs/tmp/images/' # 出力するディレクトリ
  os.makedirs(export_base, exist_ok=True)
  export_path = os.path.join(export_base, export_filename)
  
  f=io.BytesIO(content)
  im = Image.open(f)
  im.convert('L').save(export_path)
  
  return export_path

# COMMAND ----------

# MAGIC %md #### 3.2 DataframeにUDFを適用する!

# COMMAND ----------

display( 
  grayscaled_df.withColumn('output_path', export_as_jpg('path', 'grayscaled_content'))
)

# COMMAND ----------

# MAGIC %md ### 4. 出力ファイルを確認する

# COMMAND ----------

# MAGIC %fs ls /tmp/images/

# COMMAND ----------

# MAGIC %sh
# MAGIC file /dbfs/tmp/images/Browse2frame0000_proc.jpg

# COMMAND ----------

# MAGIC %md ### 5. 環境のクリーンアップ

# COMMAND ----------

# MAGIC %fs rm -r /dbfs/tmp/images/
