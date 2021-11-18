# Databricks notebook source
# MAGIC %md
# MAGIC # Vessel Tracking - Markov
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
# MAGIC In this notebook, we want to leverage the information we learned earlier by looking at common maritime routes. Specifically, we want to model these routes using historical data in order to simulate maritime traffic across the US. Using big data analytics, port authorities can better regulate inbound traffic and reduce long queues at anchorage, resulting in cost benefits for industry stakeholders and major reduction in carbon emission, *Long queues at anchorage being a major safety and environmental issue [source](https://www.portstrategy.com/news101/port-operations/planning-and-design/port-queues)*. Another application could help sea carriers be more agile and improve their operational resilience by better optimizing routes vs. economic value. As reported on [financial times](https://www.ft.com/content/65fe4650-5d90-41bc-8025-4ac81df8a5e4) *carriers have taught themselves a valuable lesson* during COVID-19 pandemic, parking up ships, sending vessels on longer journeys and cancelling hundreds of sailings. Using this framework, a cargo operator can find the shortest path from its actual location to a given destination whilst maximizing its business value.
# MAGIC 
# MAGIC <img src="https://www.portstrategy.com/__data/assets/image/0015/190122/varieties/carousel2.jpg" width=500>
# MAGIC <br>
# MAGIC [source](www.portstrategy.com)
# MAGIC <br>
# MAGIC 
# MAGIC [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) have prolific usage in mathematics. They are widely employed in economics, game theory, communication theory, genetics and finance. They arise broadly in statistical specially Bayesian statistics and information-theoretical contexts. When it comes real-world problems, they are used to postulate solutions to study cruise control systems in motor vehicles, queues or lines of customers arriving at an airport, exchange rates of currencies, etc. In this notebook, we introduce the use of Markov Chains to model maritime traffic between US ports as a steady flow traffic. This will set the foundations for extensive modelling in next notebook.
# MAGIC 
# MAGIC ### Dependencies
# MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. 

# COMMAND ----------

# DBTITLE 0,Install python libraries
# MAGIC %python
# MAGIC dbutils.library.installPyPI('networkx')
# MAGIC dbutils.library.installPyPI("keplergl")
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 0,Extra step to render KeplerGl vis.
# MAGIC %python
# MAGIC from keplergl import KeplerGl
# MAGIC import tempfile
# MAGIC 
# MAGIC def html_patch(self):
# MAGIC   """This is a patch to make Kepler work well within a Databricks Notebook"""
# MAGIC   # temp file isn't cleaned up on purpose, but it could be if that's desired
# MAGIC   (_, tmp) = tempfile.mkstemp() 
# MAGIC   self.save_to_html(file_name=tmp)
# MAGIC   with open(tmp, "r") as f:
# MAGIC     # This additional script is necessary to fix the height of the widget because peler doesn't embed well.
# MAGIC     # It mutates the containing document directly. The height parameter to keplergl.KeplerGl will be the initial 
# MAGIC     # height of the result iframe. 
# MAGIC     return f.read() + f"""<script>
# MAGIC       var targetHeight = "{self.height or 600}px";
# MAGIC       var interval = window.setInterval(function() {{
# MAGIC         if (document.body && document.body.style && document.body.style.height !== targetHeight) {{
# MAGIC           document.body.style.height = targetHeight;
# MAGIC         }}
# MAGIC       }}, 250);</script>""";
# MAGIC setattr(KeplerGl, '_repr_html_', html_patch)

# COMMAND ----------

# DBTITLE 0,Install R libraries
# MAGIC %r
# MAGIC install.packages('circlize')

# COMMAND ----------

# MAGIC %md
# MAGIC Since we aggregated data points into trips, it become easy to extract the maritime traffic between 2 distinct ports. In a markov context, a port will be defined as a state, and the transition between 2 states will be categorized by a trip, answering questions like "*What is the probability of sailing to New York city when originating from Miami?*".

# COMMAND ----------

# DBTITLE 0,Aggregate maritime routes
# MAGIC %sql
# MAGIC -- We create uniquely identifiable names as we realised port name not being unique (e.g. Portland Maine and Portland Oregon)
# MAGIC CREATE OR REPLACE TEMPORARY VIEW routes AS
# MAGIC SELECT 
# MAGIC   CONCAT(orgPortName, ' [', orgPortId, ']') AS src,
# MAGIC   CONCAT(dstPortName, ' [', dstPortId, ']') AS dst,
# MAGIC   COUNT(1) AS total
# MAGIC FROM esg.cargos_trips 
# MAGIC GROUP BY 
# MAGIC   src, 
# MAGIC   dst;
# MAGIC   
# MAGIC CACHE TABLE routes;

# COMMAND ----------

# MAGIC %md
# MAGIC We can appreciate how global our network is since each transition may lead to further branches / ramifications although we may expect at least 3 disconnected areas (west coast, east coast and great lakes areas). What is the probability of a ship to be in Portland Oregon after 2-3 trips in the west coast area? We use a [circos](http://circos.ca/) visualization to better understand the 2nd, 3rd, etc. levels of connections.

# COMMAND ----------

# DBTITLE 1,Traffic is a densely connected network
# MAGIC %r
# MAGIC library(SparkR)
# MAGIC library(circlize)
# MAGIC 
# MAGIC df <- collect(sql("SELECT src, dst, total FROM routes"))
# MAGIC 
# MAGIC from = df[[1]]
# MAGIC to = df[[2]]
# MAGIC values = df[[3]]
# MAGIC 
# MAGIC mat = matrix(0, nrow = length(unique(from)), ncol = length(unique(to)))
# MAGIC rownames(mat) = unique(from)
# MAGIC colnames(mat) = unique(to)
# MAGIC for(i in seq_along(from)) mat[from[i], to[i]] = values[i]
# MAGIC 
# MAGIC grid.col <- setNames(rainbow(length(unlist(dimnames(mat)))), union(rownames(mat), colnames(mat)))
# MAGIC par(mar = c(0, 0, 0, 0), mfrow = c(1, 1))
# MAGIC 
# MAGIC chordDiagram(mat, annotationTrack = "grid", preAllocateTracks = 1, grid.col = grid.col)
# MAGIC circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
# MAGIC   xlim = get.cell.meta.data("xlim")
# MAGIC   ylim = get.cell.meta.data("ylim")
# MAGIC   sector.name = get.cell.meta.data("sector.index")
# MAGIC   circos.text(mean(xlim), ylim[1] + .1, sector.name, facing = "clockwise", niceFacing = TRUE, adj = c(0, 0.5), cex = 0.3)
# MAGIC }, bg.border = NA)
# MAGIC 
# MAGIC circos.clear()

# COMMAND ----------

# DBTITLE 1,Build markov transitions
from pyspark.sql import functions as F
import pandas as pd

# We ensure we do not have a "terminus port" that will be translated as a state probability of 0
tied_loose_ends = spark.read.table("routes").select(F.col("dst").alias("src")).distinct() \
  .join(spark.read.table("routes"), ["src"], "left_outer") \
  .withColumn("dst", F.when(F.col("dst").isNull(), F.col("src")).otherwise(F.col("dst"))) \
  .withColumn("total", F.when(F.col("total").isNull(), F.lit(1)).otherwise(F.col("total")))

# Our state define port at trip 0, the transition is the probability to reach any other port at trip 1
markov_df = tied_loose_ends.toPandas().pivot(index='src', columns='dst', values='total').fillna(0)

# Ensure matrix is nxn
index = markov_df.index.union(markov_df.columns)
markov_df = markov_df.reindex(index=index, columns=index, fill_value=0)

# normalize to get transition state probability
markov_df = markov_df.div(markov_df.sum(axis=1), axis=0)
transition_matrix = markov_df.to_numpy()
markov_df

# COMMAND ----------

# MAGIC %md
# MAGIC Suppose there is a physical or mathematical system that has K possible states and at any one time, the system is in one and only one of its K states, we can create an input vector V representing a state K (P(k) = 1) at a time t. Using a simple matrix multiplication, our system will tell us the probability of reaching any other state at t+1. In mathematics, such a process is called [random walk](https://en.wikipedia.org/wiki/Random_walk). In our context, given an initial state (i.e. port) of e.g. New York City, where would a ship statistically be after n random walks (i.e. after completing n trips)? In the example below, we create an infinite number of simulations to identify stationary probabilities, that is, to understand ship locations after enough time at sea. 

# COMMAND ----------

# import json

# # Our state define port at trip 0, the transition is the probability to reach any other port at trip 1
# markov_df2 = tied_loose_ends.toPandas().pivot(index='src', columns='dst', values='total').fillna(0)

# # Ensure matrix is nxn
# index2 = markov_df2.index.union(markov_df2.columns)
# markov_df2 = markov_df2.reindex(index=index2, columns=index2, fill_value=0)

# json_str = json.dumps(markov_df2.to_numpy().tolist())
# with open("/tmp/circos-edges.json", "w") as f:
#   f.write(json_str)

# cols = list(markov_df.columns)

# import random

# def gen_color():
#   r = lambda: random.randint(0,255)
#   return '#%02X%02X%02X' % (r(),r(),20)

# print(pd.DataFrame([(col.split(" ")[0], gen_color()) for col in cols], columns=["name", "color"]).to_csv(index=False))

# COMMAND ----------

# DBTITLE 1,Avoid the ship "blackholes"
import numpy as np

# Create an input state vector as place of origin. Retrieve vector ID for each of these port names
port_index = list(index)
newYork = port_index.index('NewYork [134]')
fortPierce = port_index.index('FortPierce [94]') 
matagorda = port_index.index('MatagordaHarbor [3]')

# Starting from NYC, where would a ship statistically be after enough days at sea
state_vector = np.zeros(shape=(1, transition_matrix.shape[0]))
state_vector[0][newYork] = 1

# Update state vector for each simulated trip
# We run enough walks in order to find stationary probability
data = []
walks = 15000
for i in np.arange(0, walks):
  state_vector = np.dot(state_vector, transition_matrix)
  if (i % 50 == 0):
    data.append([i, 'matagorda', state_vector[0][matagorda]])
    data.append([i, 'fortPierce', state_vector[0][fortPierce]])

display(pd.DataFrame(data, columns=['i', 'port', 'probability']))

# COMMAND ----------

# MAGIC %md
# MAGIC Starting from NYC, our model indicates that with enough time, any ship would enventually reach either MatagordaHarbor (60%) or FortPierce (40%), the probability to reach any other destination becoming zero. This oddity (and potential drawback) of our model can be explained as follows. Although we have enough points to model shipment of goods from one port to another (15,000 trips), some least popular ports have no observed trips originating from (these were destination only observations). Even if the initial probability of reaching these ports were tiny, the observed chances of leaving those are null (no observations), hence acting as statistical "blackholes" no one can escape from. Naturally, given an infinite number of trips, the chance to reach one of these black holes become closer to 1.
# MAGIC 
# MAGIC In order to circumvent this oddity, we introduce a non null probability of erratic behavior, so that at any point in time, a ship captain may decide to change their routes to a random location. This additional probability will help model erratic behavior as well as acting as an escape route from those black holes. This is known in the markovian literature as a "teleport" variable that contributed to Google successful algorithm, [Page rank](https://en.wikipedia.org/wiki/PageRank). Unlike the simple random walk where transitions occur only to neighboring nodes, each transition is now accompanied by a "cointoss" that with probability d may "teleport" the random walk onto other nodes of the graph (or in this case its adjacency matrix)

# COMMAND ----------

# DBTITLE 1,Avoiding blackholes with erratic behaviors
import numpy as np
import random

# Create an input state vector as place of origin
newYork = port_index.index('NewYork [134]')
state_vector = np.zeros(shape=(1,markov_df.shape[0]))
state_vector[0][newYork] = 1

# probability d may "teleport" the random walk
teleport = 0.001

# Update state vector for each simulated trip
walks = 5
for i in np.arange(0, walks):
  toss = random.uniform(0, 1)
  if(toss <= teleport):
    # erratic behavior
    state_vector = np.zeros(shape=(1,markov_df.shape[0]))
    state_vector[0][random.randint(0, markov_df.shape[0] - 1)] = 1
    
  state_vector = np.dot(state_vector, transition_matrix)

# Return probability distribution of destination port after N trips
final_state = pd.DataFrame(state_vector[0], columns=['p'])
final_state['port'] = index
final_state  = final_state[final_state['p'] > 0.01]
final_state = final_state.sort_values(by='p', ascending=False)
display(final_state)

# COMMAND ----------

# MAGIC %md
# MAGIC As represented above, a ship owner is no longer (statistically) cursed to reach (and never escape from) Matagorda or Fort Pierce, resulting in a model that is much closer to real life scenario. However, an improvement of our model could be to model erratic behavior only with realistic destination (in close proximity). We do not expect a vessel operating in westcoast to "teleport" in the NYC harbor - nor would we want to recommend that trip. This would require some legwork to find [K nearest neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) and only "teleport" to one of these.

# COMMAND ----------

# MAGIC %md
# MAGIC One could apply this process to understand the statistical location of hazardous ships, such as vessels bearing a flag of convenience as covered in previous notebook. This would help identify areas with hightened environmental and safety concern. As per example below, we can easily generate what could be a statistically plausible journal log for any ship originating from New York City. 

# COMMAND ----------

# DBTITLE 1,Generate a random journal log
import numpy as np
import random
import re

# Create an input state vector as place of origin
newYork = port_index.index('NewYork [134]')
state_vector = np.zeros(shape=(1,markov_df.shape[0]))
state_vector[0][newYork] = 1

def getId(name):
  return int(re.findall(r"\d+", name)[0])

# probability d may "teleport" the random walk
# another way to look at teleport would be to model market volatility (low vs. high demand)
teleport = 0.001

# Randomly select one of the non zero proability for each simulated trip
walks = 10
trips = []
last_port = "NewYork [134]"
for i in np.arange(1, walks + 1):
  
  toss = random.uniform(0, 1)
  if(toss <= teleport):
    # erratic behavior
    ind = random.randint(0, markov_df.shape[0] - 1)
  else:
    # retrieve probability for each state
    pvals = np.dot(state_vector, transition_matrix)[0]
    # randomly select one
    ind = np.where(np.random.multinomial(1,pvals))[0][0]
  
  # Create a new state vector
  state_vector = np.zeros(shape=(1,markov_df.shape[0]))
  new_port = port_index[ind]
  
  # Append to journal log
  if(last_port != ""):
    trips.append([i - 1, getId(last_port), getId(new_port), toss <= teleport])
  
  # And sail!
  last_port = new_port
  state_vector[0][ind] = 1 

# Return series of N random trips
random_log = pd.DataFrame(trips, columns=['trip', 'orgPortId', 'dstPortId', 'erratic'])
random_log_df = spark.createDataFrame(random_log)

trips_df = spark \
  .read \
  .table("esg.cargos_trips") \
  .join(random_log_df, ['orgPortId', 'dstPortId']) \
  .groupBy("trip", "orgPortName", "dstPortName") \
  .agg(F.round(F.avg("duration")).alias("hours"), F.round(F.avg("distance")).alias("distance")) \
  .orderBy(F.asc("trip"))

display(trips_df)

# COMMAND ----------

# MAGIC %md
# MAGIC As we assume that the number of observations between 2 ports is positively correlated with the economic activity between these 2 cities, a high probability of reaching a port is therefore function of an higher economic value (there are more chance a cargo will find a new contract quicker), we can wonder what would be the most appropriate path to reach 2 distinct cities. For the purpose of a demo, we want to find the (economically viable) [shortest path](https://en.wikipedia.org/wiki/Shortest_path_problem) between 2 cities using networkx (graph capability) built on top of our adjacency matrix. Although the shortest path would be to directly sail towards a city, the probability distribution can tell cargo operators how to better optimize trips whilst covering as many deals as possible.

# COMMAND ----------

# DBTITLE 0,Kepler configuration
config={
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "v7j8nm",
          "type": "point",
          "config": {
            "dataId": "data",
            "label": "org",
            "color": [
              18,
              147,
              154
            ],
            "columns": {
              "lat": "orgLat",
              "lng": "orgLon",
              "altitude": None
            },
            "isVisible": True,
            "visConfig": {
              "radius": 43.3,
              "fixedRadius": False,
              "opacity": 0.8,
              "outline": False,
              "thickness": 2,
              "strokeColor": None,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": True
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
            "colorField": None,
            "colorScale": "quantile",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        },
        {
          "id": "8lawczr",
          "type": "point",
          "config": {
            "dataId": "data",
            "label": "dst",
            "color": [
              221,
              178,
              124
            ],
            "columns": {
              "lat": "dstLat",
              "lng": "dstLon",
              "altitude": None
            },
            "isVisible": True,
            "visConfig": {
              "radius": 41.8,
              "fixedRadius": False,
              "opacity": 0.8,
              "outline": False,
              "thickness": 2,
              "strokeColor": None,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": True
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
            "colorField": None,
            "colorScale": "quantile",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        },
        {
          "id": "n0nxpm9",
          "type": "arc",
          "config": {
            "dataId": "data",
            "label": "org -> dst arc",
            "color": [
              146,
              38,
              198
            ],
            "columns": {
              "lat0": "orgLat",
              "lng0": "orgLon",
              "lat1": "dstLat",
              "lng1": "dstLon"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "thickness": 2,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "sizeRange": [
                0,
                10
              ],
              "targetColor": None
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
            "colorField": None,
            "colorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "data": [
              {
                "name": "trip",
                "format": None
              },
              {
                "name": "orgPort",
                "format": None
              },
              {
                "name": "orgLat",
                "format": None
              },
              {
                "name": "orgLon",
                "format": None
              },
              {
                "name": "dstPort",
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
      "latitude": 36.58526529456187,
      "longitude": -86.7861492702887,
      "pitch": 50,
      "zoom": 4.944925131945315,
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

# DBTITLE 1,Shortest economically viable path
import numpy
import networkx as nx
import pandas as pd
from keplergl import KeplerGl
import re

# Convert adjacency matrix to graph
G=nx.from_pandas_adjacency(markov_df)
source = 'Albany [110]'
target = 'FortPierce [94]'

def getId(name):
  return int(re.findall(r"\d+", name)[0])

org = source
orgId = getId(org)

trips = []
for trip, dst in enumerate(nx.shortest_path(G,source=source,target=target)):
  dstId = getId(dst)
  trips.append([trip, orgId, dstId])
  org = dst
  orgId = dstId
  
# Enrich trip with geo coordinates
pdf = pd.DataFrame(trips, columns=['trip', 'orgPortId', 'dstPortId'])

routes = pdf.merge(ports, left_on="orgPortId", right_on="id")
routes = routes.drop(["orgPortId", "id"], axis=1)
routes = routes.rename(columns={"port": "orgPort", "latitude": "orgLat", "longitude": "orgLon"})

routes = routes.merge(ports, left_on="dstPortId", right_on="id")
routes = routes.drop(["dstPortId", "id"], axis=1)
routes = routes.rename(columns={"port": "dstPort", "latitude": "dstLat", "longitude": "dstLon"})
 
KeplerGl(height=800, data={'data': routes}, config=config)

# COMMAND ----------

# DBTITLE 1,Next steps...
# MAGIC %md
# MAGIC A direct improvement of our model would be to factor additional variables, such as weather and seasonality. In fact, an anomaly in our approach was detected where no historical data was found between Sault Ste Marie and Duluth between January and March. This apparent oddity could be certainly explained by the fact great lakes are mostly frozen in winter, so recommending such a route would not necessarily be appropriate.

# COMMAND ----------

# MAGIC %md
# MAGIC ### How to use this framework?
# MAGIC 
# MAGIC <img src="https://boardmember.com/wp-content/uploads/2020/06/AdobeStock_240658805.jpg" width=200>

# COMMAND ----------

# DBTITLE 1,Minimize disruption...
# MAGIC %md
# MAGIC Ship owners can improve their operation resilience by better reacting to emerging threats or hazardous events in real time whilst minimizing business disruption. FT estimates that "[...]*250,000 people are believed still to be marooned*" as authorities have prevented seafarers disembarking on grounds of infection risk. Using this framework, a cargo operator could for instance bring its crew back home safely by finding the shortest route that maximizes business value from its actual location to its port of origin, hence minimizing business impact of a major crisis

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
# MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
# MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
# MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
# MAGIC ---
