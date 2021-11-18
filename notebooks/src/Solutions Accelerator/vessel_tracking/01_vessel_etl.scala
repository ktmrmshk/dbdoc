// Databricks notebook source
// MAGIC %md
// MAGIC # Vessel Tracking - ETL
// MAGIC 
// MAGIC *The benefits of Environmental, Social and Governance (ESG) is well understood across the financial service industry, but the benefits of ESG goes beyond sustainable investments. What recent experience has taught us is that high social values and good governance emerged as key indicators of resilience throughout the COVID-19 pandemic. Large retailers that already use ESG to monitor the performance of their supply chain have been able to leverage this information to better navigate the challenges of global lockdowns, ensuring a constant flow of goods and products to communities. As reported in an [article](https://corpgov.law.harvard.edu/2020/08/14/the-other-s-in-esg-building-a-sustainable-and-resilient-supply-chain/) from Harvard Law School Forum on Corporate Governance, [...] companies that invest in [ESG] also benefit from competitive advantages, faster recovery from disruptions.*
// MAGIC 
// MAGIC ---
// MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
// MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
// MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
// MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
// MAGIC ---
// MAGIC <antoine.amend@databricks.com>

// COMMAND ----------

// MAGIC %md
// MAGIC ## Context
// MAGIC 
// MAGIC In this notebook, we download AIS data (Automatic Identification System) for every vessel sailing over the US coastline. Data is available yearly as an archive from [https://coast.noaa.gov](https://coast.noaa.gov) website. We simply download CSV data as-is and store all cargo related data (`vessel type = 70`) onto a Delta table. We demonstrate how one could visualise data using [KeplerGL](https://kepler.gl/) library and set the foundation of vessel tracking using H3 library for geospatial analytics.
// MAGIC 
// MAGIC ### Dependencies
// MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. In addition to below python libraries that can be installed programmatically when not using a ML runtime, one needs to install `com.uber:h3:3.6.3` maven library for geospatial capabilities

// COMMAND ----------

// DBTITLE 0,Install python libraries
// MAGIC %python
// MAGIC dbutils.library.installPyPI('beautifulsoup4')
// MAGIC dbutils.library.installPyPI("keplergl")
// MAGIC dbutils.library.installPyPI("geopandas")
// MAGIC dbutils.library.restartPython()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Download cargo data
// MAGIC <img src="https://cdn.onlinewebfonts.com/svg/img_260551.png" width=50/>
// MAGIC Note that this process is time consuming (100Gb), so you may consider downloading only a few files first.

// COMMAND ----------

// DBTITLE 0,Create scratch space on dbfs mountpoint
dbutils.fs.mkdirs("/tmp/vessels")

// COMMAND ----------

// DBTITLE 0,Create database
// MAGIC %sql
// MAGIC CREATE DATABASE IF NOT EXISTS esg

// COMMAND ----------

// DBTITLE 0,Download raw data (CAUTION - long process)
// MAGIC %sh
// MAGIC # see: https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2018/index.html
// MAGIC # we download data to dbfs:// mountpoint (/dbfs) 
// MAGIC cd /dbfs/tmp/vessels
// MAGIC wget -np -r -nH -L --cut-dirs=3 https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2018/ > /dev/null 2>&1
// MAGIC 
// MAGIC # unpack all archives
// MAGIC for FILE in `ls -1rt *.zip`; do
// MAGIC   echo $FILE
// MAGIC   unzip $FILE > /dev/null 2>&1
// MAGIC done

// COMMAND ----------

// DBTITLE 1,Downloaded vessel data
display(dbutils.fs.ls("/tmp/vessels"))

// COMMAND ----------

// DBTITLE 1,Save data to Delta Lake
import org.apache.spark.sql.types._

val schema = StructType(
  List(
    StructField("MMSI",IntegerType,true), 
    StructField("BaseDateTime",TimestampType,true), 
    StructField("LAT",DoubleType,true), 
    StructField("LON",DoubleType,true), 
    StructField("SOG",DoubleType,true), 
    StructField("COG",DoubleType,true), 
    StructField("Heading",DoubleType,true), 
    StructField("VesselName",StringType,true), 
    StructField("IMO",StringType,true), 
    StructField("CallSign",StringType,true), 
    StructField("VesselType",IntegerType,true), 
    StructField("Status",IntegerType,true), 
    StructField("Length",IntegerType,true), 
    StructField("Width",IntegerType,true), 
    StructField("Draft",DoubleType,true), 
    StructField("Cargo",IntegerType,true), 
    StructField("TranscieverClass",StringType,true)
  )
)

spark.read
  .option("header", "true")
  .option("badRecordsPath", "/tmp/ais_invalid") // bad records to quarantine
  .schema(schema)
  .csv("/tmp/vessels/AIS_ASCII_by_UTM_Month/2018")
  .filter(col("VesselType") === 70) // Only select cargos
  .filter(year(col("BaseDateTime")) === 2018)
  .filter(col("Status").isNotNull)
  .select(
    col("CallSign").as("callSign"),
    col("MMSI").as("mmsi"),
    col("VesselName").as("vesselName"),
    col("BaseDateTime").as("timestamp"),
    col("LAT").as("latitude"),
    col("LON").as("longitude"),
    col("SOG").as("sog"),
    col("Heading").as("heading"),
    col("Status").as("status")
  )
  .write.format("delta")
  .mode("overwrite")
  .saveAsTable("esg.cargos")

display(spark.read.table("esg.cargos"))

// COMMAND ----------

// MAGIC %md
// MAGIC As reported below, we have access to about 200 million for 2018 alone and only cargo related information. Our next notebook will aim at sessionizing these timestamp / locations points into well defined trips.

// COMMAND ----------

// DBTITLE 1,500m records for 2018 alone
// MAGIC %sql
// MAGIC SELECT weekofyear(`timestamp`) AS week, COUNT(*) AS total FROM esg.cargos
// MAGIC GROUP BY weekofyear(`timestamp`)
// MAGIC ORDER BY 1 ASC

// COMMAND ----------

// MAGIC %md
// MAGIC ### Data visualisation using Kepler GL
// MAGIC In this section, we want to demonstrate the use of advanced geospatial visualisations to better understand and appreciate the complexity of the use case we are tackling. In order to reduce the number of points to visualize (500 million data points), and in preparation for extensive geospatial analytics in later notebook, we introduce the use of H3 as a grid system to group data points to. Developped by [Uber](https://eng.uber.com/h3/), this Hexagonal Hierarchical Spatial Index (aka H3) is used to analyze large spatial data sets, partitioning areas of the Earth into identifiable grid cells as per image below.
// MAGIC 
// MAGIC <img src="https://1fykyq3mdn5r21tpna3wkdyi-wpengine.netdna-ssl.com/wp-content/uploads/2018/06/image12.png" width=300>
// MAGIC <br>
// MAGIC [source](https://eng.uber.com/h3/)

// COMMAND ----------

// DBTITLE 0,Geospatial utilites
import com.uber.h3core.H3Core
import com.uber.h3core.util.GeoCoord
import com.uber.h3core.LengthUnit
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._
import org.apache.spark.sql.functions._

// We create a User defined function to convert a point into polygon given a resolution
// Given a point to encode (latitude and longitude) and a resolution
// We return the hexadecimal representation of the corresponding H3 polygon
val toH3 = udf((lat: Double, lon: Double, res: Int) => {
  require(res <= 12 && res > 0, "Resolution should be positive integer < 12")
  val h3 = H3Core.newInstance()
  val h3Long = h3.geoToH3(lat, lon, res)
  f"$h3Long%X"
})

// We fill a predefined radius with as many poylgons as required given a resolution
// Given a point to encode (latitude and longitude), a resolution and a radius in Km
// We return all H3 polygons fiting our circle, encoded in their hexadecimal representations
val toH3Ring = udf((lat: Double, lon: Double, radiusKm: Int, res: Int) => {
  require(res <= 12 && res > 0, "Resolution should be positive integer < 12")
  require(radiusKm > 0, "Radius should be positive integer")
  val h3 = H3Core.newInstance()
  val origin = h3.geoToH3(lat, lon, res)
  val radius = Math.floor(radiusKm / (h3.edgeLength(res, LengthUnit.km) * 2))
  h3.kRing(origin, radius.toInt).map(h3Long => f"$h3Long%X")
})

// We register both UDFs so that we make our code portable between python and scala (via SQL)
spark.udf.register("h3", toH3)
spark.udf.register("h3_ring", toH3Ring)

// COMMAND ----------

// DBTITLE 0,Extra steps to render visualisation
// MAGIC %python
// MAGIC from keplergl import KeplerGl
// MAGIC import tempfile
// MAGIC 
// MAGIC def html_patch(self):
// MAGIC   """This is a patch to make Kepler work well within a Databricks Notebook"""
// MAGIC   # temp file isn't cleaned up on purpose, but it could be if that's desired
// MAGIC   (_, tmp) = tempfile.mkstemp() 
// MAGIC   self.save_to_html(file_name=tmp)
// MAGIC   with open(tmp, "r") as f:
// MAGIC     # This additional script is necessary to fix the height of the widget because peler doesn't embed well.
// MAGIC     # It mutates the containing document directly. The height parameter to keplergl.KeplerGl will be the initial 
// MAGIC     # height of the result iframe. 
// MAGIC     return f.read() + f"""<script>
// MAGIC       var targetHeight = "{self.height or 600}px";
// MAGIC       var interval = window.setInterval(function() {{
// MAGIC         if (document.body && document.body.style && document.body.style.height !== targetHeight) {{
// MAGIC           document.body.style.height = targetHeight;
// MAGIC         }}
// MAGIC       }}, 250);</script>""";
// MAGIC setattr(KeplerGl, '_repr_html_', html_patch)

// COMMAND ----------

// DBTITLE 0,Visualisation configuration
// MAGIC %python
// MAGIC config = {
// MAGIC   "version": "v1",
// MAGIC   "config": {
// MAGIC     "visState": {
// MAGIC       "filters": [],
// MAGIC       "layers": [
// MAGIC         {
// MAGIC           "id": "ud0x28",
// MAGIC           "type": "hexagonId",
// MAGIC           "config": {
// MAGIC             "dataId": "data",
// MAGIC             "label": "h3",
// MAGIC             "color": [
// MAGIC               18,
// MAGIC               147,
// MAGIC               154
// MAGIC             ],
// MAGIC             "columns": {
// MAGIC               "hex_id": "h3"
// MAGIC             },
// MAGIC             "isVisible": True,
// MAGIC             "visConfig": {
// MAGIC               "opacity": 0.8,
// MAGIC               "colorRange": {
// MAGIC                 "name": "Global Warming",
// MAGIC                 "type": "sequential",
// MAGIC                 "category": "Uber",
// MAGIC                 "colors": [
// MAGIC                   "#5A1846",
// MAGIC                   "#900C3F",
// MAGIC                   "#C70039",
// MAGIC                   "#E3611C",
// MAGIC                   "#F1920E",
// MAGIC                   "#FFC300"
// MAGIC                 ]
// MAGIC               },
// MAGIC               "coverage": 1,
// MAGIC               "enable3d": False,
// MAGIC               "sizeRange": [
// MAGIC                 0,
// MAGIC                 500
// MAGIC               ],
// MAGIC               "coverageRange": [
// MAGIC                 0,
// MAGIC                 1
// MAGIC               ],
// MAGIC               "elevationScale": 5
// MAGIC             },
// MAGIC             "hidden": False,
// MAGIC             "textLabel": [
// MAGIC               {
// MAGIC                 "field": None,
// MAGIC                 "color": [
// MAGIC                   255,
// MAGIC                   255,
// MAGIC                   255
// MAGIC                 ],
// MAGIC                 "size": 18,
// MAGIC                 "offset": [
// MAGIC                   0,
// MAGIC                   0
// MAGIC                 ],
// MAGIC                 "anchor": "start",
// MAGIC                 "alignment": "center"
// MAGIC               }
// MAGIC             ]
// MAGIC           },
// MAGIC           "visualChannels": {
// MAGIC             "colorField": {
// MAGIC               "name": "count",
// MAGIC               "type": "integer"
// MAGIC             },
// MAGIC             "colorScale": "quantile",
// MAGIC             "sizeField": None,
// MAGIC             "sizeScale": "linear",
// MAGIC             "coverageField": None,
// MAGIC             "coverageScale": "linear"
// MAGIC           }
// MAGIC         }
// MAGIC       ],
// MAGIC       "interactionConfig": {
// MAGIC         "tooltip": {
// MAGIC           "fieldsToShow": {
// MAGIC             "data": [
// MAGIC               {
// MAGIC                 "name": "h3",
// MAGIC                 "format": None
// MAGIC               },
// MAGIC               {
// MAGIC                 "name": "count",
// MAGIC                 "format": None
// MAGIC               }
// MAGIC             ]
// MAGIC           },
// MAGIC           "compareMode": False,
// MAGIC           "compareType": "absolute",
// MAGIC           "enabled": True
// MAGIC         },
// MAGIC         "brush": {
// MAGIC           "size": 0.5,
// MAGIC           "enabled": False
// MAGIC         },
// MAGIC         "geocoder": {
// MAGIC           "enabled": False
// MAGIC         },
// MAGIC         "coordinate": {
// MAGIC           "enabled": False
// MAGIC         }
// MAGIC       },
// MAGIC       "layerBlending": "normal",
// MAGIC       "splitMaps": [],
// MAGIC       "animationConfig": {
// MAGIC         "currentTime": None,
// MAGIC         "speed": 1
// MAGIC       }
// MAGIC     },
// MAGIC     "mapState": {
// MAGIC       "bearing": 0,
// MAGIC       "dragRotate": False,
// MAGIC       "latitude": 44.70626352772753,
// MAGIC       "longitude": -101.49221877172097,
// MAGIC       "pitch": 0,
// MAGIC       "zoom": 2.1845063353220064,
// MAGIC       "isSplit": False
// MAGIC     },
// MAGIC     "mapStyle": {
// MAGIC       "styleType": "dark",
// MAGIC       "topLayerGroups": {},
// MAGIC       "visibleLayerGroups": {
// MAGIC         "label": True,
// MAGIC         "road": True,
// MAGIC         "border": False,
// MAGIC         "building": True,
// MAGIC         "water": True,
// MAGIC         "land": True,
// MAGIC         "3d building": False
// MAGIC       },
// MAGIC       "threeDBuildingColor": [
// MAGIC         9.665468314072013,
// MAGIC         17.18305478057247,
// MAGIC         31.1442867897876
// MAGIC       ],
// MAGIC       "mapStyles": {}
// MAGIC     }
// MAGIC   }
// MAGIC }

// COMMAND ----------

// DBTITLE 1,Display maritime traffic
// MAGIC %python
// MAGIC from pyspark.sql import functions as F
// MAGIC from keplergl import KeplerGl 
// MAGIC import pandas as pd
// MAGIC 
// MAGIC df = spark \
// MAGIC   .read \
// MAGIC   .table("esg.cargos") \
// MAGIC   .groupBy(F.expr("h3(latitude, longitude, 5)").alias("h3")) \
// MAGIC   .count() \
// MAGIC   .toPandas()
// MAGIC 
// MAGIC KeplerGl(height=600, data={'data': df}, config=config)

// COMMAND ----------

// MAGIC %md
// MAGIC The above picture helps us appreciate the complexity of the use case we are addressing. Although martitime routes are pseudo defined, most of the traffic originates from / or at destination outside of the US (therefore cannot be fully captured using US coastline data). These points will be later categorized as trips where only trips originating from the US, at destination to US will be kept

// COMMAND ----------

// MAGIC %md
// MAGIC ## Download port information
// MAGIC As our study focuses on US coastal data only, the first approach towards data cleansing is to find possible ports of origin / destination using reference data we could find on the internet (https://www.searoutes.com). We use `BeautifulSoup` python library to scrape that information as a structured table. Finally, please note that port name may be duplicate (we learned at our expense that we had 2 Portland in our dataset, Portlan Oregon and Portland Maine)

// COMMAND ----------

// DBTITLE 1,Download ports in the US
// MAGIC %python
// MAGIC import re
// MAGIC import pandas as pd
// MAGIC import json
// MAGIC import requests
// MAGIC from bs4 import BeautifulSoup
// MAGIC 
// MAGIC # Scrape all information from HTML
// MAGIC url_base = 'https://www.searoutes.com/country-ports/United-States'
// MAGIC x = requests.get(url_base)
// MAGIC soup = BeautifulSoup(x.text)
// MAGIC scripts = soup.findAll('script')
// MAGIC 
// MAGIC for script in scripts:
// MAGIC   if("window.preloadedData" in str(script)):
// MAGIC     lines = str(script).split("\n")[2:-2]
// MAGIC     text = ''.join(lines)
// MAGIC     text = text.replace("countryData:", "")
// MAGIC     text = text.replace(" ", "")
// MAGIC     a_json = json.loads(text)
// MAGIC     break
// MAGIC 
// MAGIC ports = []
// MAGIC for port in a_json['ports'].items():
// MAGIC   ports.append([port[1]['name'], port[1]['lat'], port[1]['lon']])
// MAGIC 
// MAGIC # Create a well defined structure
// MAGIC ports_df = pd.DataFrame(ports, columns=['port', 'latitude', 'longitude'])
// MAGIC ports_df['id'] = ports_df.index
// MAGIC 
// MAGIC # Store port information as a delta table
// MAGIC spark.createDataFrame(ports_df).write.format("delta").mode("overwrite").saveAsTable("esg.ports")
// MAGIC display(ports_df)

// COMMAND ----------

// MAGIC %md
// MAGIC For each port, we assign well defined H3 polygon as a catchment area (a possible radius of few kilometers) since we do not expect ships to be anchored at an exact location. As reported in next notebook, this technique will help us categorize trips using a simple JOIN query instead of compute intensive geospatial "point in polygon".

// COMMAND ----------

// DBTITLE 0,Visualisation configuration
// MAGIC %python
// MAGIC config={
// MAGIC   "version": "v1",
// MAGIC   "config": {
// MAGIC     "visState": {
// MAGIC       "filters": [],
// MAGIC       "layers": [
// MAGIC         {
// MAGIC           "id": "hxbv4zo",
// MAGIC           "type": "hexagonId",
// MAGIC           "config": {
// MAGIC             "dataId": "data",
// MAGIC             "label": "h3",
// MAGIC             "color": [
// MAGIC               18,
// MAGIC               147,
// MAGIC               154
// MAGIC             ],
// MAGIC             "columns": {
// MAGIC               "hex_id": "h3"
// MAGIC             },
// MAGIC             "isVisible": True,
// MAGIC             "visConfig": {
// MAGIC               "opacity": 0.8,
// MAGIC               "colorRange": {
// MAGIC                 "name": "Global Warming",
// MAGIC                 "type": "sequential",
// MAGIC                 "category": "Uber",
// MAGIC                 "colors": [
// MAGIC                   "#5A1846",
// MAGIC                   "#900C3F",
// MAGIC                   "#C70039",
// MAGIC                   "#E3611C",
// MAGIC                   "#F1920E",
// MAGIC                   "#FFC300"
// MAGIC                 ]
// MAGIC               },
// MAGIC               "coverage": 0.8,
// MAGIC               "enable3d": True,
// MAGIC               "sizeRange": [
// MAGIC                 0,
// MAGIC                 500
// MAGIC               ],
// MAGIC               "coverageRange": [
// MAGIC                 0,
// MAGIC                 1
// MAGIC               ],
// MAGIC               "elevationScale": 5
// MAGIC             },
// MAGIC             "hidden": False,
// MAGIC             "textLabel": [
// MAGIC               {
// MAGIC                 "field": None,
// MAGIC                 "color": [
// MAGIC                   255,
// MAGIC                   255,
// MAGIC                   255
// MAGIC                 ],
// MAGIC                 "size": 18,
// MAGIC                 "offset": [
// MAGIC                   0,
// MAGIC                   0
// MAGIC                 ],
// MAGIC                 "anchor": "start",
// MAGIC                 "alignment": "center"
// MAGIC               }
// MAGIC             ]
// MAGIC           },
// MAGIC           "visualChannels": {
// MAGIC             "colorField": {
// MAGIC               "name": "port",
// MAGIC               "type": "string"
// MAGIC             },
// MAGIC             "colorScale": "ordinal",
// MAGIC             "sizeField": None,
// MAGIC             "sizeScale": "linear",
// MAGIC             "coverageField": None,
// MAGIC             "coverageScale": "linear"
// MAGIC           }
// MAGIC         }
// MAGIC       ],
// MAGIC       "interactionConfig": {
// MAGIC         "tooltip": {
// MAGIC           "fieldsToShow": {
// MAGIC             "data": [
// MAGIC               {
// MAGIC                 "name": "port",
// MAGIC                 "format": None
// MAGIC               },
// MAGIC               {
// MAGIC                 "name": "h3",
// MAGIC                 "format": None
// MAGIC               }
// MAGIC             ]
// MAGIC           },
// MAGIC           "compareMode": False,
// MAGIC           "compareType": "absolute",
// MAGIC           "enabled": True
// MAGIC         },
// MAGIC         "brush": {
// MAGIC           "size": 0.5,
// MAGIC           "enabled": False
// MAGIC         },
// MAGIC         "geocoder": {
// MAGIC           "enabled": False
// MAGIC         },
// MAGIC         "coordinate": {
// MAGIC           "enabled": False
// MAGIC         }
// MAGIC       },
// MAGIC       "layerBlending": "normal",
// MAGIC       "splitMaps": [],
// MAGIC       "animationConfig": {
// MAGIC         "currentTime": None,
// MAGIC         "speed": 1
// MAGIC       }
// MAGIC     },
// MAGIC     "mapState": {
// MAGIC       "bearing": 24,
// MAGIC       "dragRotate": True,
// MAGIC       "latitude": 37.497916619501076,
// MAGIC       "longitude": -75.59395297636873,
// MAGIC       "pitch": 50,
// MAGIC       "zoom": 6.856249015020366,
// MAGIC       "isSplit": False
// MAGIC     },
// MAGIC     "mapStyle": {
// MAGIC       "styleType": "dark",
// MAGIC       "topLayerGroups": {},
// MAGIC       "visibleLayerGroups": {
// MAGIC         "label": True,
// MAGIC         "road": True,
// MAGIC         "border": False,
// MAGIC         "building": True,
// MAGIC         "water": True,
// MAGIC         "land": True,
// MAGIC         "3d building": False
// MAGIC       },
// MAGIC       "threeDBuildingColor": [
// MAGIC         9.665468314072013,
// MAGIC         17.18305478057247,
// MAGIC         31.1442867897876
// MAGIC       ],
// MAGIC       "mapStyles": {}
// MAGIC     }
// MAGIC   }
// MAGIC }

// COMMAND ----------

// DBTITLE 1,Visualise port catchment area
// MAGIC %python
// MAGIC from pyspark.sql import functions as F
// MAGIC # we crate a 40km radius made of H3 polygon around port location
// MAGIC # this will become a catchment area to sessionize trips
// MAGIC from keplergl import KeplerGl 
// MAGIC df = spark.read.table("esg.ports") \
// MAGIC   .withColumn("h3_ring", F.expr("h3_ring(latitude, longitude, 40, 5)")) \
// MAGIC   .select(F.col("port"), F.explode("h3_ring").alias("h3")) \
// MAGIC   .toPandas()
// MAGIC 
// MAGIC KeplerGl(height=500, data={'data': df}, config=config)

// COMMAND ----------

// MAGIC %md
// MAGIC Although we enriched ports with catchment areas, we can observe overlaps for ports in close vicinity. We will need to take this information into consideration when sessionizing points into trips. A simple approach would be to assign ports having the highest [jaccard](https://en.wikipedia.org/wiki/Jaccard_index) index (number of polygons matching its catchment area)

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
// MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
// MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
// MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
// MAGIC ---
