# Databricks notebook source
# MAGIC %md
# MAGIC # Vessel Tracking - predict
# MAGIC 
# MAGIC *The benefits of Environmental, Social and Governance (ESG) is well understood across the financial service industry, but the benefits of ESG goes beyond sustainable investments. What recent experience has taught us is that high social values and good governance emerged as key indicators of resilience throughout the COVID-19 pandemic. Large retailers that already use ESG to monitor the performance of their supply chain have been able to leverage this information to better navigate the challenges of global lockdowns, ensuring a constant flow of goods and products to communities. As reported in an [article](https://corpgov.law.harvard.edu/2020/08/14/the-other-s-in-esg-building-a-sustainable-and-resilient-supply-chain/) from Harvard Law School Forum on Corporate Governance, [...] companies that invest in [ESG] also benefit from competitive advantages, faster recovery from disruptions.*
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
# MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
# MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
# MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC 
# MAGIC In previous <a href="$./03_vessel_markov">notebook</a>, we introduced the use of [Markov Chains](https://en.wikipedia.org/wiki/Markov_chain) to better understand the maritime traffic between 2 different ports as a steady state flow. In this notebook, we want to go one level more granular and understand the transition from one geographic location to another. Given a port of origin and a current location, we would like to know were a ship is statistically heading to. By "random walking" these sequences recusrively, we will be able to predict the destination of any given ship alongside its estimated time of arival (ETA). 
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. In addition to below python libraries that can be installed programmatically when not using a ML runtime, one needs to install `com.uber:h3:3.6.3` maven library for geospatial capabilities

# COMMAND ----------

# DBTITLE 0,Install libraries
dbutils.library.installPyPI("keplergl")
dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 0,Import libraries
import numpy as np
import pandas as pd
import os

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StringType, LongType, StructType, StructField
from keplergl import KeplerGl 

import mlflow.pyfunc

np.seterr(divide = 'ignore') 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building Markov chains
# MAGIC 
# MAGIC The first consideration to bear in mind is the memory-less nature of Markov chain. To put it another way, the current location does not carry information about previous steps (where a ship originates from). Instead of considering each location as a transition state, one could consider a vector (or a tuple of 2 locations) as a second-order memory transition state. This, however, would dramatically increase the complexity of our transition matrix. Instead, we create a dedicated model for each originating port (about 200 for US coastal dataset).
# MAGIC 
# MAGIC The second consideration is around the granularity used in defining our transition states. As we work with geospatial coordinates, a high granularity of our H3 polygons would create a sparse (and therefore large) transition matrix whilst a lower granularity would prevent us from running actual predictions. After different attempts, we estimated that a H3 granularity of 4 would be a fine balance between probability density and accurate predictions (approx. 20km radius prediction). Please refer to previous notebooks for more information on H3.

# COMMAND ----------

# DBTITLE 0,Collect H3 polygon across journeys
# MAGIC %scala
# MAGIC import com.uber.h3core.H3Core
# MAGIC 
# MAGIC val toH3 = udf((lat: Double, lon: Double) => {
# MAGIC   val h3 = H3Core.newInstance()
# MAGIC   val hex = h3.geoToH3(lat, lon, 4)
# MAGIC   f"$hex%X"
# MAGIC })
# MAGIC 
# MAGIC // Register scala function as UDF so that it can be used in Pyspark (via F.expr)
# MAGIC spark.udf.register("h3", toH3)

# COMMAND ----------

# MAGIC %md
# MAGIC In order to run accurate predictions, we first ensure that the destination for each trip is the actual port location (i.e. its h3 polygon) rather than approximate point. As experienced in previous <a href="$./03_vessel_markov">notebook</a>, we also ensure to "close our black holes", that is to link all destinations states (i.e. ports) to themselves (the probability of staying on these states become 1). If not, the probability distribution would be 0 resulting in `NaN` matrix

# COMMAND ----------

# DBTITLE 1,Build markov transitions
# Create a mapping between port ID and H3 polygon, exposing this mapping as a UDF for convenience
ports_df = spark.read.table("esg.ports").withColumn("h3", F.expr("h3(latitude, longitude)")).toPandas()
ports_h3_dict = spark.sparkContext.broadcast(dict(zip(ports_df['id'], ports_df['h3'])))

@udf("string")
def port_to_h3(id):
  return ports_h3_dict.value[id]

# Create polygon for each data point
points = spark.read.table("esg.cargos_points").withColumn("state", F.expr("h3(latitude, longitude)"))

# We create clear states where point at origin and destinations are replaced with exact port locations
# We use a Window partitioning to understand the first and last record of each trip, replacing with actual port H3
states = spark.read.table("esg.cargos_trips") \
    .withColumn("orgPortH3", port_to_h3(F.col("orgPortId"))) \
    .withColumn("dstPortH3", port_to_h3(F.col("dstPortId"))) \
    .join(points, ["tripId"]) \
    .withColumn("asc", F.dense_rank().over(Window.partitionBy("tripId").orderBy(F.asc("timestamp")))) \
    .withColumn("dsc", F.dense_rank().over(Window.partitionBy("tripId").orderBy(F.desc("timestamp")))) \
    .withColumn("state", F.when(F.col("asc") == 1, F.col("orgPortH3")).otherwise(F.col("state"))) \
    .withColumn("state", F.when(F.col("dsc") == 1, F.col("dstPortH3")).otherwise(F.col("state"))) \
    .select("tripId", "timestamp", "state", "orgPortId", "dstPortId")

# We create our transitions by linking each state to its next location
# We ensure that destination states (i.e. ports) are linked to themselves to ensure markov properties
transitions = states \
  .withColumn("next", F.lead("state", 1).over(Window.partitionBy("tripId").orderBy(F.asc("timestamp")))) \
  .withColumn("next", F.when(F.col("next").isNull(), F.col("state")).otherwise(F.col("next"))) \
  .filter((F.col("next") != F.col("state")) | (F.col("dsc") == 1)) \
  .groupBy(F.col("orgPortId"), F.col("dstPortId"), F.col("tripId"), F.col("state").alias("src"), F.col("next").alias("dst")) \
  .agg(
    F.sum(F.lit(1)).alias("count"),
    F.max(F.col("timestamp")).alias("timestamp")
  )

display(transitions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate our hypothesis
# MAGIC In order to validate our hypothesis, we first start with a simple example of predicting trips originating from Miami. We split all trips originating from Miami as a training and testing set. Given the transition states extracted from our training set, can we accurately predict the destination of each test sample?

# COMMAND ----------

# DBTITLE 0,Retrieve all journeys originating from Miami
m_df = transitions.filter(F.col("orgPortId") == 21).toPandas()

# COMMAND ----------

# DBTITLE 0,Retrieve list of possible destination ports
m_ports = m_df.merge(ports_df, left_on="dst", right_on="h3")[['h3', 'port']]
m_ports_names = dict(zip(m_ports['h3'], m_ports['port']))

# COMMAND ----------

# DBTITLE 0,Build a transition matrix for every trip originating from Miami
def test_train_flag(df):
  
  # extract list of all trips   
  trips = df["tripId"].unique()
  ratio = int(0.1 * len(trips))
  np.random.seed(seed=42)
  np.random.shuffle(trips)
  
  # keep 10% for testing, 90% for training
  df['testing'] = df['tripId'].isin(trips[:ratio])
  return df

def build_markov(training_df):
  
  # aggregate transitions using pivot table  
  markov_df = training_df.pivot_table(index = "src", columns = "dst", values = "count", aggfunc = "sum").fillna(0)
  index = markov_df.index.union(markov_df.columns)
  markov_df = markov_df.reindex(index=index, columns=index, fill_value=0)

  # normalize to get transition state probability
  markov_df = markov_df.div(markov_df.sum(axis=1), axis=0)
  return markov_df  

# Split between training and testing set
m_df = test_train_flag(m_df)
m_tr = m_df[m_df['testing'] == False]
m_ts = m_df[m_df['testing'] == True]

# Building markov chains
m_markov = build_markov(m_tr)
m_matrix = m_markov.to_numpy()
m_index = list(m_markov.index)

# COMMAND ----------

# DBTITLE 1,Display some test journeys
display(
  m_ts \
    .merge(ports_df, left_on="dstPortId", right_on="id").rename(columns={"port": "dstPort"}) \
    .groupby(["tripId", "dstPort"]).count().index.to_frame(index=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC As random walk is an iterative process and is theoretically unbounded, we may have to define a success criteria. As reported in previous notebook, we know that destination nodes (linked to themselves) would eventually result in stationary probabilities (what we defined as "black holes" in previous notebook). With too little walks, we may not reach any destination port and may only predict 1 or 2 polygons away from actual location. With too many walks, we may not bring additional improvement to the returned probability distribution. Eventually, we want to stop our random walk process when the probability distribution remains unchanged (within a predefined threshold). For that purpose, we use [Bhattacharyya](https://en.wikipedia.org/wiki/Bhattacharyya_distance) coeffiencient as a distance measure between 2 probability distribution. When distance is lower than `epsilon=0.001`, we consider our probability distribution to be stationary and stop our random walk process (capped to 1000 simulations)

# COMMAND ----------

# DBTITLE 1,Probability distribution throughout a given journey
# We take one random trip from our test data
m_states = list(m_ts[m_ts['tripId'] == "382-99721"].src)

# We run `max_walks` random walks from a given state
max_walks = 1000

# We stop walking through the transition state if the probability distribution does not change by more than `epsilon`
epsilon = 0.001

def bhattacharyya(p, q):
  bc=np.sum(np.sqrt(p*q))
  b=-np.log(bc)
  return b

data = []
# We will try to predict destination at every step of the journey, starting from the first observed location
for i, state in enumerate(m_states):
  
  try:
    # Given the observed location, we create a state vector...
    start = m_index.index(state) 
    state_vector = np.zeros(shape=(1, m_markov.shape[0]))
    state_vector[0][start] = 1
    
    # ... that we update for each simulated trip
    for walk in np.arange(0, max_walks):
      
      new_state_vector = np.dot(state_vector, m_matrix)
      distance = bhattacharyya(state_vector, new_state_vector)
      state_vector = new_state_vector
      # we carry on random walks until the distribution is considered stationary
      if(distance < epsilon):
        break
        
    # We extract probability for each known destination
    for h3 in m_ports_names.keys():
      port_idx = m_index.index(h3) 
      port = m_ports_names[h3]
      p = state_vector[0][port_idx]
      data.append([i, state, walk, port, p])
    
  except:
    # we may never have observed this state before, so ignore and move on
    pass
  
# Store all predictions to a pandas dataframe
predictions = pd.DataFrame(data, columns=['i', 'state', 'walks', 'port', 'probability'])
top_ports = list(predictions.groupby('port').max().sort_values(by='probability', ascending=False).head(5).index)
top_predictions = predictions[predictions['port'].isin(top_ports)]
display(top_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that the most probable destination was Wilmington (30% of chances) until the ship started to head east, moving toward the NewYork / Philadelphia route (probabilities were similar). At step #40 (around 80% completion), it become obvious that our ship was heading towards NYC (as the probability of heading towards Philadelphia drammatically dropped to zero). 

# COMMAND ----------

# DBTITLE 0,Extra step to render KeplerGl vis.
from keplergl import KeplerGl
import tempfile

def html_patch(self):
  """This is a patch to make Kepler work well within a Databricks Notebook"""
  # temp file isn't cleaned up on purpose, but it could be if that's desired
  (_, tmp) = tempfile.mkstemp() 
  self.save_to_html(file_name=tmp)
  with open(tmp, "r") as f:
    # This additional script is necessary to fix the height of the widget because peler doesn't embed well.
    # It mutates the containing document directly. The height parameter to keplergl.KeplerGl will be the initial 
    # height of the result iframe. 
    return f.read() + f"""<script>
      var targetHeight = "{self.height or 600}px";
      var interval = window.setInterval(function() {{
        if (document.body && document.body.style && document.body.style.height !== targetHeight) {{
          document.body.style.height = targetHeight;
        }}
      }}, 250);</script>""";
setattr(KeplerGl, '_repr_html_', html_patch)

# COMMAND ----------

# DBTITLE 0,Create Kepler config
config={
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "cwjam3l",
          "type": "hexagonId",
          "config": {
            "dataId": "data",
            "label": "h3",
            "color": [
              18,
              147,
              154
            ],
            "columns": {
              "hex_id": "h3"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 1,
              "colorRange": {
                "name": "ColorBrewer OrRd-6",
                "type": "sequential",
                "category": "ColorBrewer",
                "colors": [
                  "#fef0d9",
                  "#fdd49e",
                  "#fdbb84",
                  "#fc8d59",
                  "#e34a33",
                  "#b30000"
                ]
              },
              "coverage": 0.73,
              "enable3d": True,
              "sizeRange": [
                0,
                500
              ],
              "coverageRange": [
                0,
                1
              ],
              "elevationScale": 5
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": {
              "name": "probability",
              "type": "real"
            },
            "colorScale": "quantile",
            "sizeField": {
              "name": "walks",
              "type": "integer"
            },
            "sizeScale": "log",
            "coverageField": None,
            "coverageScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "data": [
              {
                "name": "h3",
                "format": None
              },
              {
                "name": "count",
                "format": None
              },
              {
                "name": "probability",
                "format": None
              },
              {
                "name": "walks",
                "format": None
              }
            ]
          },
          "compareMode": False,
          "compareType": "absolute",
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 24,
      "dragRotate": True,
      "latitude": 31.69836459468918,
      "longitude": -76.18423426075957,
      "pitch": 50,
      "zoom": 5.00064547092282,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "dark",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": True,
        "water": True,
        "land": True,
        "3d building": False
      },
      "threeDBuildingColor": [
        9.665468314072013,
        17.18305478057247,
        31.1442867897876
      ],
      "mapStyles": {}
    }
  }
}

# COMMAND ----------

# DBTITLE 1,Visualize trip and prediction
# display trajectory as sequence of H3 polygons
df = spark.read.table("esg.cargos_points") \
  .filter(F.col("tripId") == "382-99721") \
  .withColumn("h3", F.expr("h3(latitude, longitude)")) \
  .groupBy("h3").count() \
  .toPandas()

# we enrich each polygon with the actual probability of reaching NYC
df_merge = df.merge(predictions[predictions['port'] == "NewYork"], left_on='h3', right_on='state')
df_merge = df_merge[['h3', 'count', 'probability', 'walks']]
KeplerGl(height=800, data={'data': df_merge}, config=config)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented in above figure, we observe the probability of reaching NYC increase over time. The height of pylogon represents the number of walks required to consider our probability distribution stationary. It is obvious that as we reach our destination, the probability increase and less random walks are required (much more confidence in our probabilities). 
# MAGIC 
# MAGIC After multiple tests, we have been able to observe an evident drawback in our model. There are multiple ports densily packed around specific regions. An example would be Houston TX area with Freeport, Houston, Galveston, Matagorda ports, all located within a 50-100km radius, and all sharing a same inbound route pattern (overall direction pointing towards Houston). As a consequence, the most popular port would shadow its least popular neighbours resulting in low probability distribution and apparent low accuracy. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extending our approach to all originating ports
# MAGIC Although our approach is statistically valid, an improvement would be to predict larger geographical area (such as major cities for example) instead of specific ports, factor weather and seasonality variables or to bring additional data such as cargo size / draft to better understand a port capacity to welcome cargos of different sizes. For the purpose of that demo, we want to extend our baseline model globally by creating a specific markov chain for each originating port in parallel, leveraging `pandasUDF` paradigm

# COMMAND ----------

# DBTITLE 0,Create scratch space where models will be serialized
try:
  dbutils.fs.rm("/tmp/vessel_models", True)
except:
  pass

dbutils.fs.mkdirs("/tmp/vessel_models")

# COMMAND ----------

# DBTITLE 1,Training models in parallel, serializing Markov chains to DBFS
schema = StructType(
  [
    StructField('orgPortId', LongType(), True), 
    StructField('name', StringType(), True), 
    StructField('location', StringType(), True)
  ]
)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def train_model(key, df):
  
  orgPortId = key[0]
  
  # we'll store our serialize matrix on dbfs
  temp_dir = "/dbfs/tmp/vessel_models" 
    
  # create our markov chains
  markov = build_markov(df)
  matrix = markov.to_numpy()
  index = list(markov.index)
  
  # save matrix
  np.save(temp_dir + "/mat_{}.npy".format(orgPortId), matrix)
  
  # save index
  np.save(temp_dir + "/ind_{}.npy".format(orgPortId), index)
  
  data = [
    [orgPortId, "ind_{}".format(orgPortId), temp_dir + "/ind_{}.npy".format(orgPortId)],
    [orgPortId, "mat_{}".format(orgPortId), temp_dir + "/mat_{}.npy".format(orgPortId)]
  ]
  
  # return pointer to serialize objects
  return pd.DataFrame(data=data, columns=["orgPortId", "name", "location"])
  
# train multiple models in parallel, one for each originating port
models = transitions.groupBy("orgPortId").apply(train_model).toPandas()
display(models)

# COMMAND ----------

# MAGIC %md
# MAGIC Storing respectives indices and matrices for each port is not enough, we also need to serialize our ports locations (as H3 polygons) to predict actual destinations.

# COMMAND ----------

# DBTITLE 0,Serialize ports dictionary
ports_dict = dict(zip(ports_df['h3'], ports_df['port']))
np.save('/dbfs/tmp/vessel_models/ports.npy', ports_dict) 

# COMMAND ----------

# MAGIC %md
# MAGIC Our strategy would be to create a "uber model" that can redirect incoming request to its dedicated markov chain given a provided port of origin and a current H3 location. We'll be using a `pyFunc` object that can deserialize markov chains to run specific predictions. Note that we consider our model lazy and only load the required matrices, caching them for further use.

# COMMAND ----------

# DBTITLE 0,Registering pyfunc to dynamically call the relevant model
class VesselPrediction(mlflow.pyfunc.PythonModel):
  
  import numpy as np
  
  mats = {}
  inds = {}
  
  def _bhattacharyya(self, p, q):
    bc=np.sum(np.sqrt(p*q))
    b=-np.log(bc)
    return b
  
  def load_context(self, context):
    # deserializing our dictionary of H3 to Port name
    self.ports = np.load(context.artifacts['ports'],allow_pickle='TRUE').item()
    
  def predict(self, context, df):
  
    data = []
    
    # for each incoming record...
    for i, row in df.iterrows():
      
      # we retrieve the corresponding model...
      orgPortId = row.orgPortId
      if (orgPortId not in self.inds):
        # we load / cache model if not retrieved yet
        self.mats[orgPortId] = np.load(context.artifacts['mat_{}'.format(orgPortId)])
        self.inds[orgPortId] = list(np.load(context.artifacts['ind_{}'.format(orgPortId)]))
        
      mat = self.mats[orgPortId]
      ind = self.inds[orgPortId]  
      
      # starting from a given H3 state...
      if (row.state in ind):
        
        # we initialize an input vector...
        start_id = ind.index(row.state)
        state_vector = np.zeros(shape=(1, mat.shape[0]))
        state_vector[0][start_id] = 1
        
        # and random walk through our transition matrix...
        for walk in np.arange(0, 100):
          new_state_vector = np.dot(state_vector, mat)
          distance = self._bhattacharyya(state_vector, new_state_vector)
          state_vector = new_state_vector
          
          # stopping when probability becomes stationary...
          if (distance < 0.01):
            break 

        # we retrieve the highest probability...
        p_id = np.argmax(state_vector[0])
        dst = ind[p_id]
        
        # that hopefully corresponds to a given port of destination...
        if(dst in self.ports):
          data.append([orgPortId, row.state, self.ports[dst], state_vector[0][p_id]])
        else:
          # predicted location is not a known port...
          data.append([orgPortId, row.state, None, 1.0])
      else:
        # provided state is not part of trained model...
        data.append([orgPortId, row.state, None, 1.0])
        
    # return the predicted values
    return pd.DataFrame(data, columns=["orgPortId", "state", "predicted", "probability"])

# COMMAND ----------

# MAGIC %md
# MAGIC With a pyfunc model defined, we register on `mlflow` by attaching all artifacts generated at earlier step. This includes all markov chains generated for each originating port Id.

# COMMAND ----------

# DBTITLE 0,Attaching all artifacts to our pyfunc model
with mlflow.start_run(run_name='vessel_predict'):
  
  artifacts = dict(zip(models['name'], models['location']))
  artifacts['ports'] = '/dbfs/tmp/vessel_models/ports.npy'
  
  mlflow.pyfunc.log_model(
    'model', 
    python_model=VesselPrediction(), 
    artifacts=artifacts
  )
  
  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run predictions
# MAGIC With our model stored on `mlflow`, we can easily predict destinations of incoming records. All we need is a port of origin (known) and an actual location. We will return a possible destination with its probability or `None` if the actual location was not captured by our training set. Note that the following are for demonstration only since we use same data used in earlier steps. We would like to understand how our model would perform for different ports of origin and at different time throughout their respective journeys. 

# COMMAND ----------

# DBTITLE 1,Create a test dataset to simulate inbound trips
# create a simple dictionary of port Id to names that we expose through UDF for convenience
ports_df = spark.read.table("esg.ports").toPandas()
ports_name_dict = spark.sparkContext.broadcast(dict(zip(ports_df['id'], ports_df['port'])))

@udf("string")
def port_to_name(id):
  return ports_name_dict.value[id]

# retrieve initial data points and rank order within a trip
ranked_points = spark \
  .read \
  .table("esg.cargos_points") \
  .withColumn("state", F.expr("h3(latitude, longitude)")) \
  .withColumn("rank", F.dense_rank().over(Window.partitionBy("tripId").orderBy("timestamp")))

# count how many records we have for each trip
ranked_points_max = ranked_points \
  .groupBy("tripId") \
  .agg(F.max(F.col("rank")).alias("n"))

# compute journey completion for each data point
all_points = ranked_points.join(ranked_points_max, ["tripId"]) \
  .withColumn("completion", F.col("rank") / F.col("n")) \
  .select("tripId", "state", "completion")
  
# randomly select 10000 data points to simulate inbound traffic
inbound_points = all_points.sample(10000.0 / all_points.count())

# create our test data set that shows inbound points with actual destination
test_data = inbound_points \
  .join(spark.read.table("esg.cargos_trips"), ["tripId"]) \
  .withColumn("origin", port_to_name(F.col("orgPortId"))) \
  .withColumn("destination", port_to_name(F.col("dstPortId"))) \
  .select("orgPortId", "state", "completion", "origin", "destination") \
  .toPandas()

display(test_data)

# COMMAND ----------

# DBTITLE 1,Predicting destination
# we load our pyfunc model from MLFlow and predict inbound trips location
model = mlflow.pyfunc.load_model("runs:/{}/model".format(run_id))
model_preds = model.predict(test_data)

# we retrieve the actual destination and current trip completion, dropping records we could not predict
model_preds['destination'] = test_data['destination']
model_preds['completion'] = test_data['completion']
model_preds = model_preds.merge(ports_df[['id', 'port']], left_on="orgPortId", right_on="id").drop(["id", "orgPortId"], axis=1)
model_preds = model_preds.rename(columns={"port":"origin"})
model_preds = model_preds.dropna()

# we return predicted vs. actual
model_preds = model_preds[["origin", "destination", "completion", "predicted", "probability"]]
model_preds = model_preds.round(3)
display(model_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC As expected, we can observe that the returned probability increases as a trip is reaching its actual destination (positive correlation between probability and completion). We are obviously more confident about a prediction when the destination port is in sight.

# COMMAND ----------

# DBTITLE 1,Accuracy (obviously) increases with trip completion
accuracy_df = model_preds[["completion"]].copy()
accuracy_df["correct_prediction"] = model_preds["predicted"] == model_preds["destination"]
accuracy_df = accuracy_df.round(2)
display(pd.DataFrame(accuracy_df.groupby(["completion", "correct_prediction"]).size(), columns=["count"]).reset_index())

# COMMAND ----------

# DBTITLE 1,Next steps...
# MAGIC %md
# MAGIC The probability is correlated with trip completion, so is its accuracy. As vessels are completing their journeys, our model would predict destinations much more accurately. In average, our model accuracy starts to be greater than 50% when a vessel is half way through its journey. As reported earlier, this apparent low accuracy is to be taken with a pinch of salt. In fact, our predictions are globally correct and consistent, but shadowed by more popular ports in densely packed areas. To fully appreciate the predictive power of our approach, one would need to look at the actual distance between predicted vs. actual locations as a more appropriate success metric. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to use this framework?
# MAGIC 
# MAGIC <img src="https://boardmember.com/wp-content/uploads/2020/06/AdobeStock_240658805.jpg" width=200>

# COMMAND ----------

# DBTITLE 1,Detect and react...
# MAGIC %md
# MAGIC We demonstrated how companies can better understand maritime traffic of not only their own fleet but also those of their competitors. Using weather forecast, environmental or safety concerns (direction of least regulated ships), anomaly detection to detect damage, ship owners have now access to a real time lense of a high density traffic where **auxiliary correction can be made real time** to vessels approaching danger state (it may take ~20 minutes for a fully loaded large tanker to stop when heading at normal speed)

# COMMAND ----------

# DBTITLE 1,Predict and optimize...
# MAGIC %md
# MAGIC With fuel costs representing as much as 50-60% of total ship operating costs, ship owner can leverage this framework to better estimate the Energy Efficiency Operational Indicator (EEOI) of their fleet at a time t and their time of arrival (ETA). Overlayed with predicted traffic at destination, they can **establish a strategic plan to better optimize fuel consumption** such as reducing sailing speed, re-routing, etc to avoid long queues at anchorage. Note that without any knowledge of ship characteristics (such as hull frictional resistance or propulsion efficiency), fuel consumption could still be approximated using contextual information we have, like speed over ground, actual draft (itself a proxy of tonnage) and weather conditions [[source](https://www.sciencedirect.com/science/article/pii/S2092678220300091)]

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
# MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
# MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
# MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
# MAGIC ---
