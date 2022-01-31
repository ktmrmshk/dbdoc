# Databricks notebook source
# MAGIC %md # DataFrame内の画像ファイル(JPG)をExportする

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

# MAGIC %md ### 2. 画像ファイルをExportするためのUDFを作成

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
  im.save(export_path)
  
  return export_path

# COMMAND ----------

# MAGIC %md ### 3. DataframeにUDFを適用する!

# COMMAND ----------

display( 
  image_df.withColumn('added_col', export_as_jpg('path', 'content'))
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
