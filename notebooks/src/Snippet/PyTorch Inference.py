# Databricks notebook source
# MAGIC %md 
# MAGIC # Spark環境を用いたPytorchの推定の分散並列化
# MAGIC (Original Notebook: [Distributed model inference using PyTorch](https://docs.databricks.com/_static/notebooks/deep-learning/pytorch-images.html))
# MAGIC 
# MAGIC 
# MAGIC このノートブックでは、[torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50)のResNet-50モデルと画像ファイルを入力データとして、PyTorchを使って分散モデル推論を行う方法を紹介しています。
# MAGIC 
# MAGIC 以下のステップで説明します。
# MAGIC 
# MAGIC * **1. 学習済みモデルの読み込み**
# MAGIC * **2. 推論を充てるデータの準備(Deltaテーブル)** 
# MAGIC * **3. UDF(ユーザー定義関数)を作成、推論を実施** 
# MAGIC 
# MAGIC **Note:**
# MAGIC * GPUを使用する/しないに応じて、変数`cuda`を設定してください。
# MAGIC   - GPUあり =>  `cuda = True`
# MAGIC   - GPUなし =>  `cuda = False`

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 画像ファイルの配置
# MAGIC 
# MAGIC 推論を実施する画像ファイルが以下のAzure blob storage上に配置してあるものとして説明します。
# MAGIC 
# MAGIC * ストレージアカウント: `test_storage_account123`
# MAGIC * ストレージコンテナ: `image-container`
# MAGIC * コンテナ内のpath: `/demo/daisy/xxxxxxxxx.jpg` (大量のjpgファイル群)
# MAGIC 
# MAGIC 
# MAGIC 全てのクラスタノードのpythonコードから上記のblog storage上のファイルを参照できるように、DBFSにマウントします。

# COMMAND ----------

# ストレージアカウント、コンテナ、アクセスキーの設定
storage_account = 'test_storage_account123'
storage_container = 'image-container'
access_key = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx==' # 本番環境ではsecrets機能を用いて、平文では扱わないようにする


# dbfs上の`/mnt/images`に上記のstorage containerをマウントする
# 以降、
#  * Sparkコード上からは`dbfs:/mnt/images/...`もしくは、`/mnt/images/...`で、
#  * 通常のPythonコード/shellからは`/dbfs/mnt/images/...`で
# アクセス可能になる

dbutils.fs.mount(
  source = f'wasbs://{storage_container}@{storage_account}.blob.core.windows.net/',
  mount_point = '/mnt/images',
  extra_configs = {
    f'fs.azure.account.key.{storage_account}.blob.core.windows.net': access_key
  }
)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /dbfs/mnt/images/demo/daisy/

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1. 学習済みモデルの読み込み

# COMMAND ----------

# このサンプルコードでは使用しない
cuda = False

# COMMAND ----------

import os
import shutil
import uuid
from typing import Iterator, Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader  # private API

from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType

# COMMAND ----------

#GPU or CPU?
use_cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# COMMAND ----------

# MAGIC %md
# MAGIC ResNet50 のモデル(pre-trained)をドライバーノードにロードして、その後、Broadcast変数にして、全部のクラスタノードから参照できるようにしておく。
# MAGIC 
# MAGIC **補足**: シングルコンピュートとは異なり、Sparkで分散処理するには、全てのクラスタを構成するノードが同等にデータにアクセスできるようにする必要があリます。例えば、ファイル読み込みに関しては、AWS S3やAzure Blog Storageを使用することで、どのノードからも同一のパスで同じファイルにアクセス可能になります。
# MAGIC 
# MAGIC 一方で、今回のような"モデル"オブジェクトをメモリ上にロードして、クラスタ上で利用するには、オブジェクトデータを全てのクラスタ(のメモリ上)にもロードさせる必要があります。そのための仕組みが「broadcast変数」であり、下記のコードの通り、
# MAGIC 
# MAGIC * `sc.broadcast()`で変数をbroadcastでき、
# MAGIC * 各ノード上では、上記のbroadcast変数名でアクセスできる
# MAGIC 
# MAGIC ようになります。

# COMMAND ----------

# ドライバノード(親ノード)上にモデルを読み込む
model_state = models.resnet50(pretrained=True).state_dict()

# 上記で読み込んだオブジェクトをbroadcastする
bc_model_state = sc.broadcast(model_state)

# COMMAND ----------

# この関数は、あとで、各クラスタから呼ばれる想定で定義してある
def get_model_for_eval():
  """Gets the broadcasted model."""
  model = models.resnet50(pretrained=True)
  model.load_state_dict(bc_model_state.value) # broadcast変数から値を参照している
  model.eval()
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. 推論を適用するデータの準備(SparkのDataFrame)
# MAGIC 
# MAGIC pythonコードから参照できる画像のパスのdataframeを作成します。

# COMMAND ----------

# 画像のファイルパスのリストを作る
image_dir = '/mnt/images/demo/daisy/'
img_files = []

for f in dbutils.fs.ls(image_dir):
  img_files.append( f.path.replace('dbfs:', '/dbfs') )

print(img_files)

# COMMAND ----------

# 上記で作成した画像ファイルのパスをSpark DataFrame化する
img_df = (
  spark
  .createDataFrame( img_files, 'string')
  .withColumnRenamed('value', 'path')
  .repartition(64) # クラスタのコア数の倍数にしてください。
)

display(img_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. UDF(ユーザー定義関数)を作成、推論を実施

# COMMAND ----------

# MAGIC %md
# MAGIC カスタムのPytorch Datasetを定義

# COMMAND ----------

class ImageDataset(Dataset):
  def __init__(self, paths, transform=None):
    self.paths = paths
    self.transform = transform
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    image = default_loader(self.paths[index])
    if self.transform is not None:
      image = self.transform(image)
    return image

# COMMAND ----------

# MAGIC %md
# MAGIC 推論を実施する関数(UDF)を定義。
# MAGIC 
# MAGIC * この関数の引数`paths`に上記で作成したSpark DataFrameの`path`カラムの内容が代入され、並列分散的に呼ばれる。
# MAGIC * この関数の戻り値がSparkのDataFrameのカラムとして取り出される

# COMMAND ----------

@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(paths: pd.Series) -> pd.Series:
  transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
  ])
  images = ImageDataset(paths, transform=transform)
  loader = torch.utils.data.DataLoader(images, batch_size=500, num_workers=8)
  model = get_model_for_eval()
  model.to(device)
  all_predictions = []
  with torch.no_grad():
    for batch in loader:
      predictions = list(model(batch.to(device)).cpu().numpy())
      for prediction in predictions:
        all_predictions.append(prediction)
  return pd.Series(all_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC 推論の実行!

# COMMAND ----------

# 推論の実行!
predictions_df = (
  img_df
  .select( col('path'), predict_batch_udf( col('path') ).alias("prediction") ) #<== UDFをDataframeの`path`カラムに当てている
)

# 結果のファイル書き出し(Deltaフォーマット)
( 
  predictions_df
  .write
  .format('delta')
  .mode("overwrite")
  .save('/tmp/result.delta')  # デモのため、DBFS上に書き出している。通常はblog storage上に保存する方が良い。
)

# COMMAND ----------

# MAGIC %md
# MAGIC 結果の確認

# COMMAND ----------

result_df = spark.read.format('delta').load('/tmp/result.delta')
display(result_df)

# COMMAND ----------

# spark dataframeが扱いづらい場合は、pandas dataframeに変換もできます。
p_df = result_df.toPandas()
p_df

# COMMAND ----------


