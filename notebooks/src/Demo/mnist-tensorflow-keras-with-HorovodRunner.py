# Databricks notebook source
# MAGIC %md
# MAGIC # HorovodRunnerを用いたTensorFlow/Kerasの分散Deep Learning
# MAGIC 
# MAGIC このノートブックでは、`tensorflow.keras` API を用いた分散学習のための HorovodRunner の使用方法を説明しています。
# MAGIC 最初に単一のノードでモデルをトレーニングする方法を示し、次にHorovodRunnerを使ってコードを分散トレーニング用に適応させる方法を示します。
# MAGIC このノートブックは、CPUおよびGPUクラスター上で動作します。
# MAGIC 
# MAGIC ## 要件
# MAGIC Databricks Runtime 7.0 ML以上を使用してください。
# MAGIC 
# MAGIC HorovodRunnerは、複数のワーカーを持つクラスタでのモデル学習性能を向上させるために設計されていますが、このノートブックを実行するのに複数のワーカーは必要ありません。

# COMMAND ----------

# MAGIC %md ## データを準備する関数を作成する
# MAGIC 
# MAGIC `get_dataset()`関数は学習用のデータを準備します。この関数は `rank` と `size` を引数に取るので，シングルノードと分散型のどちらの学習にも利用できます．Horovodでは、`rank`はユニークなプロセスIDで、`size`はプロセスの総数です。
# MAGIC 
# MAGIC この関数は，`keras.datasets` からデータをダウンロードし，利用可能なノードにデータを分配し，学習に必要な形状やタイプに変換します．

# COMMAND ----------

def get_dataset(num_classes, rank=0, size=1):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('MNIST-data-%d' % rank)
  x_train = x_train[rank::size]
  y_train = y_train[rank::size]
  x_test = x_test[rank::size]
  y_test = y_test[rank::size]
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  y_train = keras.utils.to_categorical(y_train, num_classes)
  y_test = keras.utils.to_categorical(y_test, num_classes)
  return (x_train, y_train), (x_test, y_test)

# COMMAND ----------

# MAGIC %md ## モデルトレーニングのための関数を作成する
# MAGIC 
# MAGIC `get_model()`関数は、`tensorflow.keras` APIを使ってモデルを定義します。このコードは[Keras MNIST convnet example](https://keras.io/examples/vision/mnist_convnet/)を参考にしています。

# COMMAND ----------

def get_model(num_classes):
  from tensorflow.keras import models
  from tensorflow.keras import layers
  
  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=(28, 28, 1)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(num_classes, activation='softmax'))
  return model

# COMMAND ----------

# MAGIC %md ## シングルノードでトレーニングを実施

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC 以下のセルの`train()`関数は、`tensorflow.keras`を使ったシングルノードのトレーニングコードを定義しています。

# COMMAND ----------

# Specify training parameters
batch_size = 128
epochs = 2
num_classes = 10

def train(learning_rate=1.0):
  from tensorflow import keras
  
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes)
  model = get_model(num_classes)

  # Specify the optimizer (Adadelta in this example), using the learning rate input parameter of the function so that Horovod can adjust the learning rate during training
  optimizer = keras.optimizers.Adadelta(lr=learning_rate)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))
  return model

# COMMAND ----------

# MAGIC %md
# MAGIC `train()`関数を実行して，driverノードのモデルをトレーニングします．このプロセスには数分かかります．エポック毎に精度が向上していきます。

# COMMAND ----------

model = train(learning_rate=0.1)

# COMMAND ----------

# MAGIC %md 損失と精度を算出

# COMMAND ----------

_, (x_test, y_test) = get_dataset(num_classes)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=128)
print("loss:", loss)
print("accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md ## 分散型トレーニングのためのHorovodRunnerへ移行する
# MAGIC 
# MAGIC ここでは、シングルノードのコードを修正してHorovodを使用する方法を紹介します。Horovodの詳細については、[Horovod documentation](https://horovod.readthedocs.io/en/stable/)を参照してください。 
# MAGIC 
# MAGIC 
# MAGIC まず、モデルのチェックポイントを保存するディレクトリを作ります。

# COMMAND ----------

import os
import time

# Remove any existing checkpoint files
dbutils.fs.rm(("/ml/MNISTDemo/train"), recurse=True)

# Create directory
checkpoint_dir = '/dbfs/ml/MNISTDemo/train/{}/'.format(time.time())
os.makedirs(checkpoint_dir)
print(checkpoint_dir)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC The following cell shows how to modify the single-node code of the previously defined `train()` function to take advantage of distributed training.  

# COMMAND ----------

def train_hvd(checkpoint_path, learning_rate=1.0):
  
  # Import tensorflow modules to each worker
  from tensorflow.keras import backend as K
  from tensorflow.keras.models import Sequential
  import tensorflow as tf
  from tensorflow import keras
  import horovod.tensorflow.keras as hvd
  
  # Initialize Horovod
  hvd.init()

  # Pin GPU to be used to process local rank (one GPU per process)
  # These steps are skipped on a CPU cluster
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  # Call the get_dataset function you created, this time with the Horovod rank and size
  (x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
  model = get_model(num_classes)

  # Adjust learning rate based on number of GPUs
  optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())

  # Use the Horovod Distributed Optimizer
  optimizer = hvd.DistributedOptimizer(optimizer)

  model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # Create a callback to broadcast the initial variable states from rank 0 to all other processes.
  # This is required to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.
  callbacks = [
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  ]

  # Save checkpoints only on worker 0 to prevent conflicts between workers
  if hvd.rank() == 0:
      callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))

  model.fit(x_train, y_train,
            batch_size=batch_size,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2,
            validation_data=(x_test, y_test))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC これで、HorovodRunnerを使ってモデルのトレーニング作業を分担する準備が整いました。
# MAGIC 
# MAGIC 
# MAGIC HorovodRunnerのパラメータ`np`は、プロセスの数を設定します。この例では、2つのワーカーがそれぞれ1つのプロセッサを持つクラスタを使用しているので、`np=2`を設定します。(`np=1` を使用すると、HorovodRunner はドライバーノード上の単一プロセスを使用してトレーニングを行います)。
# MAGIC 
# MAGIC 処理の内部でHorovodRunnerは、Horovodフックを持つ深層学習トレーニングコードを含むPythonメソッドを受け取ります。HorovodRunner は、ドライバ上でメソッドをピクルス化し、Spark ワーカーに配信します。Horovod MPIジョブは、バリア実行モードを使用してSparkジョブとして埋め込まれます。最初のエクゼキュータは BarrierTaskContext を使って全てのタスクエクゼキュータの IP アドレスを収集し、`mpirun` を使って Horovod ジョブをトリガーします。各Python MPIプロセスは、ピクルス化されたユーザプログラムをロードし、それをデシリアライズして実行します。
# MAGIC 
# MAGIC 詳しくは[HorovodRunner API documentation](https://databricks.github.io/spark-deep-learning/#api-documentation)をご覧ください。

# COMMAND ----------

from sparkdl import HorovodRunner

checkpoint_path = checkpoint_dir + '/checkpoint-{epoch}.ckpt'
learning_rate = 0.1
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=checkpoint_path, learning_rate=learning_rate)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## HorovodRunnerでトレーニングしたモデルをロードして、推定を実施する
# MAGIC 
# MAGIC 次のコードは、HorovodRunnerでのトレーニング完了後にモデルにアクセスしてロードする方法を示しています。TensorFlowのメソッド`tf.train.latest_checkpoint()`を使って、保存された最新のチェックポイントファイルにアクセスしています。

# COMMAND ----------

import tensorflow as tf

hvd_model = get_model(num_classes)
hvd_model.compile(optimizer=tf.keras.optimizers.Adadelta(lr=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
hvd_model.load_weights(tf.train.latest_checkpoint(os.path.dirname(checkpoint_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC テストデータセットでモデルの性能を評価する。

# COMMAND ----------

_, (x_test, y_test) = get_dataset(num_classes)
loss, accuracy = hvd_model.evaluate(x_test, y_test, batch_size=128)
print("loaded model loss and accuracy:", loss, accuracy)

# COMMAND ----------

# MAGIC %md モデルを使って新しいデータの予測を行う。例えば、テストデータセットの最初の10個のオブザベーションを新しいデータの代わりに使います。

# COMMAND ----------

import numpy as np

# Use rint() to round the predicted values to the nearest integer
preds = np.rint(hvd_model.predict(x_test[0:9]))
preds
