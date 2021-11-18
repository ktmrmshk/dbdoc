// Databricks notebook source
// MAGIC %md
// MAGIC # Vessel Tracking - sessionize
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
// MAGIC Given the scale and noise of data (see <a href="$./01_vessel_etl">STAGE1</a>), this notebook is a real engineering challenge. In fact, it is the perfect example of what alternative data is. When industry introduced the concept of 3Vs (volume, velocity and variety) for big data in the 2010's era, alternative data follows all 3 of them plus a 4th one: **veracity**. The information comes at different scale, different quality, is often incomplete and highly aggressive (IOT type data). All that requires the right scientific personas (such as a Data Scientist **AND** a Computer Scientist) rather than simple ETL engineering. Afterall, the more complex a task is, higher is the rewards when generating alpha. In this notebook, we want to access all isolated data points (timestamp and geocoordinate) and "sessionize" these points into well defined trips from port A to port B. This will drastically reduce the data space to drive more descriptive and predictive modelling at later stage.
// MAGIC 
// MAGIC ### Dependencies
// MAGIC As reported in below cell, we use multiple 3rd party libraries that must be made available across Spark cluster. Assuming you are running this notebook on a Databricks cluster that does not make use of the ML runtime, you can use `dbutils.library.installPyPI()` utility to install python libraries in that specific notebook context. For java based libraries, or if you are using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load libraries to your environment. In addition to below python libraries that can be installed programmatically when not using a ML runtime, one needs to install `com.uber:h3:3.6.3` maven library for geospatial capabilities

// COMMAND ----------

// DBTITLE 0,Install libraries
// MAGIC %python
// MAGIC dbutils.library.installPyPI("keplergl")
// MAGIC dbutils.library.installPyPI("geopandas")
// MAGIC dbutils.library.restartPython()

// COMMAND ----------

// MAGIC %md
// MAGIC ## Sessionize data into trips
// MAGIC lets first try to understand the complexity of the task at hand. Besides the overall number of records, we may wonder how many records does each vessel has in order to find the most appropriate method to group points into trips

// COMMAND ----------

// DBTITLE 1,Some cargos may have up to half a million data points
// MAGIC %sql
// MAGIC SELECT mmsi, COUNT(1) AS total FROM esg.cargos
// MAGIC GROUP BY mmsi
// MAGIC ORDER BY 2 DESC

// COMMAND ----------

// DBTITLE 1,Secondary Sorting
// MAGIC %md
// MAGIC The idea is to sessionize points into trips, separated by points where vessel is no longer under command (status is 1,2,5 or 6 as per [ref](https://help.marinetraffic.com/hc/en-us/articles/203990998-What-is-the-significance-of-the-AIS-Navigational-Status-Values-)). The problem is hard. Firstly, a trip is theoretically unbound, using an unbounded partition window strategy would result in each location to be held in memory. Secondly, some vessels may exhibit half a million datapoints, sorting large lists would lead to CPU inneficiencies. Lastly, our data set is highly unbalanced, and a strategy for a given vessel may be suboptimal for others. We overcome this challenge by grouping all vessels into partitions and [secondary sorting](https://www.oreilly.com/library/view/data-algorithms/9781491906170/ch01.html) data points by timestamps. This apparent legacy pattern (famous in the MapReduce era) is still incredibly useful with highly imbalanced and massive dataset and a must have in modern data science toolbox. This approach, however, forces us to move from a `DataFrame` abstraction layer to its `RDD` representation, but don't worry, we will structure our code in well defined stages.

// COMMAND ----------

// DBTITLE 0,Retrieve cargos with the most data points
import org.apache.spark.sql.functions._

val topCargos = spark
  .read
  .table("esg.cargos")
  .groupBy("mmsi")
  .count()
  .filter(col("count") > 120000)
  .select("mmsi")
  .rdd
  .map(_.getAs[Int]("mmsi"))
  .collect()

val topCargos_B = spark.sparkContext.broadcast(topCargos)

// COMMAND ----------

// MAGIC %md
// MAGIC We have 300 cargos with more than 120k data points. In order to guarantee high parallelism with no hot spot (that is, all our spark executors will work in parallel and won't be waiting for a few of them to complete), we decide to handle these cargos separately, within dedicated partitions. The rest of cargos (having number of points between 0 and 120k) will be handled in shared partitions.

// COMMAND ----------

// DBTITLE 1,Window partition function
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import java.sql.Timestamp

// first, we want to be able to separate records for different MMSI
// so we create a simple rank function, row = 0 corresponds to the first record seen for a given ship (MMSI)
// we also retrieve the time difference between 2 successive data points so that disconnected trips may not be used (more than X hours apart)

val deltaT = udf((x1: Timestamp, x2: Timestamp) => {
  scala.util.Try((x2.getTime - x1.getTime) / 60000).getOrElse(0L)
})

val rankedCargos = spark.read.table("esg.cargos")
  .withColumn("rank", dense_rank().over(Window.partitionBy(col("mmsi")).orderBy(col("timestamp"))))
  .withColumn("nextTs", lead("timestamp", 1).over(Window.partitionBy(col("mmsi")).orderBy(col("timestamp"))))
  .withColumn("deltaT", deltaT(col("timestamp"), col("nextTs")))
  .drop("nextTs")

display(rankedCargos)

// COMMAND ----------

// DBTITLE 1,Composite key for secondary sorting
// MAGIC %md
// MAGIC A nice illustration of the secondary sort pattern is represented below. The idea is to leverage Spark Suffle by creating a custom partitioner on a composite key. The first half of the key is used to define the partition number (i.e. the MMSI) whilst its second half is used to sort each key within a given partition (i.e. timestamp). Note that in an ideal world where data can easily be kept at its highest abstraction layer (such as a `DataFrame`), one could leverage [hints](https://kb.databricks.com/data/skew-hints-in-join.html) to address this challenge of skewed partitions at scale. In our example, challenge goes much beyond a simple skew problem as we also need to sort records in their dedicated partitions (secondary sorting), hence converting our dataframe into a key value pair RDD made of `VesselKey` and `VesselAtt`
// MAGIC 
// MAGIC <img src="/files/antoine.amend/images/esg_composite_key.png" alt="logical_flow" width="500">

// COMMAND ----------

// DBTITLE 0,Create case classes
// We will be grouping all attributes within a given partition, ordered by timestamp
case class VesselAtt(
                     timestamp: Timestamp,
                     deltaT: Long,
                     latitude: Double,
                     longitude: Double,
                     heading: Double,
                     status: Int,
                     sog: Double
                     )

// We include the timestamp (ranked in previous cell) in the key for secondary sorting
case class VesselKey(
                    callSign: String,
                    mmsi: Int,
                    vesselName: String,
                    rank: Int
                    )

// Create a companion object to tell framework how to sort keys
// Records will be sorted (ascending) via their rank defined earlier (derived from timestamp)
object VesselKey {
  implicit def orderingByIdVesselPosition[A <: VesselKey] : Ordering[A] = {
    Ordering.by(fk => (fk.mmsi, fk.rank))
  }
}

// COMMAND ----------

// DBTITLE 0,Create our custom partitioner
import org.apache.spark.Partitioner

class VesselKeyPartitioner() extends Partitioner {
  
  // We start with 600 partitions (heuristic) where some will be reserved for topX cargos defined earlier
  override def numPartitions: Int = 600
  
  // We assign a specific partition for each incoming record (a VesselKey)
  override def getPartition(key: Any): Int = {
    
    val k = key.asInstanceOf[VesselKey]
    val topN = topCargos_B.value
    val sharedPartitions = numPartitions - topN.length
    
    if(topN.contains(k)) {
      // Top X cargo (more than 120K data points) are sent to their dedicated partition
      sharedPartitions + topN.indexOf(k.mmsi)
    } else {
      // Not a top X cargo, send to one of the shared partitions
      math.abs(k.mmsi % sharedPartitions)
    }
  }
}

// COMMAND ----------

// MAGIC %md
// MAGIC Within a given partition, we receive an iteration of records that were "magically" sorted by the spark framework given our strategy defined in earlier cells. The concept is to split an iteration of records into an iteration of list, where each list is defined with entry and exit criteria. The purists among yourselves will note that we never had to maintain an entire collection to memory. Our breaking condition can be anything we want as captured in our Key Value pair (e.g. vessel status or speed over ground).

// COMMAND ----------

// DBTITLE 0,Sessionization logic
// Split an iterator into multiple iterators on breaking condition (e.g. moored boat)
// We'll return an iterator of sessions, processing one given session at a time
def iterativeSplit[T](iter: Iterator[T])(breakOn: T => Boolean): Iterator[List[T]] = {
  new Iterator[List[T]] {
    def hasNext: Boolean = iter.hasNext
    def next: List[T] = {
      if (iter.hasNext) {
        val cur = iter.next() +: iter.takeWhile(!breakOn(_)).toList
        iter.dropWhile(breakOn)
        cur
      } else {
        iter.takeWhile(!breakOn(_)).toList
      }
    }
  }.withFilter(l => l.nonEmpty)
}

// COMMAND ----------

// MAGIC %md
// MAGIC Finally, here were are, properly equipped to face what initially was an insurmountable wall. We convert our dataframe to a key value pair RDD and tell Spark framework how to handle our records using `repartitionAndSortWithinPartitions` (other word for secondary sorting). We pass our custom partitioner and handle our iteration of (sorted) records within each partition (via the `mapPartitionsWithIndex` function). Finally, we split our points into sessions (i.e. trips) using the following conditions: Trips must be separated by moored / anchored status (status of 1,2,5 or 6) **OR** the MMSI has changed (new rank ordered) **OR** if subsequent points are more than 1 day apart. The last condition is key to ensure vessel departing from the US to abroad and "magically" appearing a few months later would not be considered as part of the same trip (our dataset is US coastal only).

// COMMAND ----------

// DBTITLE 0,Apply our sessionization strategy
val cargoPointsRdd = rankedCargos.rdd.map(r => {
  (
    VesselKey(
      r.getAs[String]("callSign"),
      r.getAs[Int]("mmsi"),
      r.getAs[String]("vesselName"),
      r.getAs[Int]("rank")
    ),
    VesselAtt(
      r.getAs[java.sql.Timestamp]("timestamp"),
      r.getAs[Long]("deltaT"),
      r.getAs[Double]("latitude"),
      r.getAs[Double]("longitude"),
      r.getAs[Double]("heading"),
      r.getAs[Int]("status"),
      r.getAs[Double]("sog")
    )
  )
})
// apply secondary sort to group and order points at a partition level
.repartitionAndSortWithinPartitions(new VesselKeyPartitioner())
// all points for a given partition are ordered by timestamp
.mapPartitionsWithIndex({ case (pId, it) =>
  // We split by trips separated by moored / anchored ship or if MMSI has changed (new rank ordered) or if points are more than 1d apart
  val breakOn = (kv: (VesselKey, VesselAtt)) => {Set(1, 2, 5, 6).contains(kv._2.status) || kv._1.rank == 1 || kv._2.deltaT > 24 * 60}
  iterativeSplit(it)(breakOn).zipWithIndex.flatMap({ case (session, sessionId) =>
    // We assign a trip Id using index of the partition and index of the split
    val uuid = s"${pId}-${sessionId}"
    session.dropWhile(_._2.sog == 0).map({ case (key, att) =>
      (uuid, key.callSign, key.mmsi, key.vesselName, att.timestamp, att.latitude, att.longitude, att.sog, att.heading)
    })
  })
})

// 3mn to sessionize all points into 15K trips, storing results as a delta table
val cargoPoints = cargoPointsRdd.toDF("tripId", "callSign", "mmsi", "vesselName", "timestamp", "latitude", "longitude","sog","heading")
cargoPoints.write.format("delta").mode("overwrite").saveAsTable("esg.cargos_points")

// COMMAND ----------

// DBTITLE 0,Optimize delta table for faster read
// MAGIC %sql
// MAGIC OPTIMIZE esg.cargos_points ZORDER BY (tripId)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Extracting characteristics of a trip
// MAGIC With all our sessions now defined, we can safely retrieve the distance, the duration at sea together with points at origin and destination. For that purpose, we scrape the list of known US ports to find the closest known location. Although we will be leveraging H3 to find port at origin / destination, we will compute the trip distance using the [Haversine](https://en.wikipedia.org/wiki/Haversine_formula) distance between 2 subsequents geo points (using earth curvature). 

// COMMAND ----------

// DBTITLE 0,Geo distance function
case class GeoPoint(latitude: Double, longitude: Double) 

object GeoUtils {
  // Haversine distance implementation in KM
  val AVERAGE_RADIUS_OF_EARTH = 6371
  def distance(start: GeoPoint, stop: GeoPoint): Double = {
      val latDistance = Math.toRadians(start.latitude - stop.latitude)
      val lngDistance = Math.toRadians(start.longitude - stop.longitude)
      val sinLat = Math.sin(latDistance / 2)
      val sinLng = Math.sin(lngDistance / 2)
      val a = sinLat * sinLat + (Math.cos(Math.toRadians(start.latitude)) * Math.cos(Math.toRadians(stop.latitude)) * sinLng * sinLng)
      val c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
      (AVERAGE_RADIUS_OF_EARTH * c)
  }
}

// We create a user defined function that returns distance in Km between 2 distinct points
val distance = udf((startLatitude: Double, startLongitude: Double, stopLatitude: Double, stopLongitude: Double) => {
  scala.util.Try {
    GeoUtils.distance(
      GeoPoint(startLatitude, startLongitude),
      GeoPoint(stopLatitude, stopLongitude)
    )
  }.getOrElse(0.0)
})

// COMMAND ----------

// MAGIC %md
// MAGIC A vessel MMSI may not be unique (call sign is) but shall remain consistent withing a given trip. As reported [here](https://help.marinetraffic.com/hc/en-us/articles/205220087-Which-way-is-information-on-a-vessel-s-flag-found-), we can retrieve a vessel flag given a MMSI number. By looking at how flags may change for a same vessel, we could identify vessels bearing a [flag of convenience](https://en.wikipedia.org/wiki/Flag_of_convenience), usually indicative of a poor environmental, social and governance operating model. 

// COMMAND ----------

// DBTITLE 0,Create map of MMSI <-> Flag
//https://help.marinetraffic.com/hc/en-us/articles/215699608-MIDs-Countries-and-Flags-full-table-
val mmsiFlags = """201	Albania	AL
202	Andorra	AD
203	Austria	AT
204	Portugal	PT
205	Belgium	BE
206	Belarus	BY
207	Bulgaria	BG
208	Vatican	VA
209	Cyprus	CY
210	Cyprus	CY
211	Germany	DE
212	Cyprus	CY
213	Georgia	GE
214	Moldova	MD
215	Malta	MT
216	Armenia	ZZ
218	Germany	DE
219	Denmark	DK
220	Denmark	DK
224	Spain	ES
225	Spain	ES
226	France	FR
227	France	FR
228	France	FR
229	Malta	MT
230	Finland	FI
231	Faroe Is	FO
232	United Kingdom	GB
233	United Kingdom	GB
234	United Kingdom	GB
235	United Kingdom	GB
236	Gibraltar	GI
237	Greece	GR
238	Croatia	HR
239	Greece	GR
240	Greece	GR
241	Greece	GR
242	Morocco	MA
243	Hungary	HU
244	Netherlands	NL
245	Netherlands	NL
246	Netherlands	NL
247	Italy	IT
248	Malta	MT
249	Malta	MT
250	Ireland	IE
251	Iceland	IS
252	Liechtenstein	LI
253	Luxembourg	LU
254	Monaco	MC
255	Portugal	PT
256	Malta	MT
257	Norway	NO
258	Norway	NO
259	Norway	NO
261	Poland	PL
262	Montenegro	ME
263	Portugal	PT
264	Romania	RO
265	Sweden	SE
266	Sweden	SE
267	Slovakia	SK
268	San Marino	SM
269	Switzerland	CH
270	Czech Republic	CZ
271	Turkey	TR
272	Ukraine	UA
273	Russia	RU
274	FYR Macedonia	MK
275	Latvia	LV
276	Estonia	EE
277	Lithuania	LT
278	Slovenia	SI
279	Serbia	RS
301	Anguilla	AI
303	USA	US
304	Antigua Barbuda	AG
305	Antigua Barbuda	AG
306	Curacao	CW
307	Aruba	AW
308	Bahamas	BS
309	Bahamas	BS
310	Bermuda	BM
311	Bahamas	BS
312	Belize	BZ
314	Barbados	BB
316	Canada	CA
319	Cayman Is	KY
321	Costa Rica	CR
323	Cuba	CU
325	Dominica	DM
327	Dominican Rep	DO
329	Guadeloupe	GP
330	Grenada	GD
331	Greenland	GL
332	Guatemala	GT
334	Honduras	HN
336	Haiti	HT
338	USA	US
339	Jamaica	JM
341	St Kitts Nevis	KN
343	St Lucia	LC
345	Mexico	MX
347	Martinique	MQ
348	Montserrat	MS
350	Nicaragua	NI
351	Panama	PA
352	Panama	PA
353	Panama	PA
354	Panama	PA
355	Panama	PA
356	Panama	PA
357	Panama	PA
358	Puerto Rico	PR
359	El Salvador	SV
361	St Pierre Miquelon	PM
362	Trinidad Tobago	TT
364	Turks Caicos Is	TC
366	USA	US
367	USA	US
368	USA	US
369	USA	US
370	Panama	PA
371	Panama	PA
372	Panama	PA
373	Panama	PA
374	Panama	PA
375	St Vincent Grenadines	VC
376	St Vincent Grenadines	VC
377	St Vincent Grenadines	VC
378	British Virgin Is	VG
379	US Virgin Is	VI
401	Afghanistan	AF
403	Saudi Arabia	SA
405	Bangladesh	BD
408	Bahrain	BH
410	Bhutan	BT
412	China	CN
413	China	CN
414	China	CN
416	Taiwan	TW
417	Sri Lanka	LK
419	India	IN
422	Iran	IR
423	Azerbaijan	AZ
425	Iraq	IQ
428	Israel	IL
431	Japan	JP
432	Japan	JP
434	Turkmenistan	TM
436	Kazakhstan	KZ
437	Uzbekistan	UZ
438	Jordan	JO
440	Korea	KR
441	Korea	KR
443	Palestine	PS
445	DPR Korea	KP
447	Kuwait	KW
450	Lebanon	LB
451	Kyrgyz Republic	ZZ
453	Macao	ZZ
455	Maldives	MV
457	Mongolia	MN
459	Nepal	NP
461	Oman	OM
463	Pakistan	PK
466	Qatar	QA
468	Syria	SY
470	UAE	AE
472	Tajikistan	TJ
473	Yemen	YE
475	Yemen	YE
477	Hong Kong	HK
478	Bosnia and Herzegovina	BA
501	Antarctica	AQ
503	Australia	AU
506	Myanmar	MM
508	Brunei	BN
510	Micronesia	FM
511	Palau	PW
512	New Zealand	NZ
514	Cambodia	KH
515	Cambodia	KH
516	Christmas Is	CX
518	Cook Is	CK
520	Fiji	FJ
523	Cocos Is	CC
525	Indonesia	ID
529	Kiribati	KI
531	Laos	LA
533	Malaysia	MY
536	N Mariana Is	MP
538	Marshall Is	MH
540	New Caledonia	NC
542	Niue	NU
544	Nauru	NR
546	French Polynesia	TF
548	Philippines	PH
553	Papua New Guinea	PG
555	Pitcairn Is	PN
557	Solomon Is	SB
559	American Samoa	AS
561	Samoa	WS
563	Singapore	SG
564	Singapore	SG
565	Singapore	SG
566	Singapore	SG
567	Thailand	TH
570	Tonga	TO
572	Tuvalu	TV
574	Vietnam	VN
576	Vanuatu	VU
577	Vanuatu	VU
578	Wallis Futuna Is	WF
601	South Africa	ZA
603	Angola	AO
605	Algeria	DZ
607	St Paul Amsterdam Is	XX
608	Ascension Is	IO
609	Burundi	BI
610	Benin	BJ
611	Botswana	BW
612	Cen Afr Rep	CF
613	Cameroon	CM
615	Congo	CG
616	Comoros	KM
617	Cape Verde	CV
618	Antarctica	AQ
619	Ivory Coast	CI
620	Comoros	KM
621	Djibouti	DJ
622	Egypt	EG
624	Ethiopia	ET
625	Eritrea	ER
626	Gabon	GA
627	Ghana	GH
629	Gambia	GM
630	Guinea-Bissau	GW
631	Equ. Guinea	GQ
632	Guinea	GN
633	Burkina Faso	BF
634	Kenya	KE
635	Antarctica	AQ
636	Liberia	LR
637	Liberia	LR
642	Libya	LY
644	Lesotho	LS
645	Mauritius	MU
647	Madagascar	MG
649	Mali	ML
650	Mozambique	MZ
654	Mauritania	MR
655	Malawi	MW
656	Niger	NE
657	Nigeria	NG
659	Namibia	NA
660	Reunion	RE
661	Rwanda	RW
662	Sudan	SD
663	Senegal	SN
664	Seychelles	SC
665	St Helena	SH
666	Somalia	SO
667	Sierra Leone	SL
668	Sao Tome Principe	ST
669	Swaziland	SZ
670	Chad	TD
671	Togo	TG
672	Tunisia	TN
674	Tanzania	TZ
675	Uganda	UG
676	DR Congo	CD
677	Tanzania	TZ
678	Zambia	ZM
679	Zimbabwe	ZW
701	Argentina	AR
710	Brazil	BR
720	Bolivia	BO
725	Chile	CL
730	Colombia	CO
735	Ecuador	EC
740	UK	UK
745	Guiana	GF
750	Guyana	GY
755	Paraguay	PY
760	Peru	PE
765	Suriname	SR
770	Uruguay	UY
775	Venezuela	VE""".split("\n").map(l => {
  val a = l.split("\\s+")
  (a.head.toInt, a.last)
}).toMap

// COMMAND ----------

// DBTITLE 0,Get vessel flag from MMSI
import org.apache.spark.sql.functions._

val mmsiFlags_b = spark.sparkContext.broadcast(mmsiFlags)

val getFlag = udf((mmsi: Int) => {
  val code = mmsi.toString.substring(0,3).toInt
  mmsiFlags_b.value.get(code)
})

spark.udf.register("flag", getFlag)

// COMMAND ----------

// DBTITLE 1,Aggregate trip characteristics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

val trips = spark.read.table("esg.cargos_points")
  .withColumn("s", struct(col("latitude"), col("longitude")))
  .withColumn("n", lead("s", 1).over(Window.partitionBy(col("mmsi")).orderBy(col("timestamp"))))
  // compute distance from any point to its next
  .withColumn("distance", distance(col("s.latitude"), col("s.longitude"), col("n.latitude"), col("n.longitude")))
  .groupBy("tripId", "callSign", "mmsi")
  .agg(
    first("vesselName").as("vesselName"),
    first("timestamp").as("orgTime"),
    last("timestamp").as("dstTime"),
    first(struct(col("latitude"), col("longitude"))).as("orgGeo"),
    last(struct(col("latitude"), col("longitude"))).as("dstGeo"),
    sum("distance").as("distance")
  )
  .withColumn("flag", getFlag(col("mmsi")))
  .cache()

display(trips)

// COMMAND ----------

// MAGIC %md
// MAGIC As introduced in previous (see <a href="$./01_vessel_etl">notebook</a>), one can use H3 for an efficient radius lookup, as long as they're willing to approximate the circle as a k-ring of hexagons. This is much faster than a true search via e.g. Haversine distance against each port, but less accurate. As we no longer have to run expensive point in polygon search for each location, this approach would scale for any city worldwide (see [blog](https://observablehq.com/@nrabinowitz/h3-radius-lookup)).

// COMMAND ----------

// DBTITLE 0,Converting latitudes and longitudes to H3 polygons
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

// DBTITLE 1,Index ports by multiple hexagons
// Index ports on a 40km radius
// Granularity of 5 covers radius with enough polygons
val optimalRes = 5
val portIndexDf = spark.read.table("esg.ports")
  .withColumn("hexagons", toH3Ring(col("latitude"), col("longitude"), lit(40), lit(optimalRes)))
  .withColumn("portLocation", toH3(col("latitude"), col("longitude"), lit(10)))
  .withColumnRenamed("port", "portName")
  .withColumnRenamed("id", "portId")
  .select(col("portId"), col("portName"), col("portLocation"), explode(col("hexagons")).as("h3"))
  .cache()

display(portIndexDf)

// COMMAND ----------

// DBTITLE 0,Time difference function
import java.sql.Timestamp
import org.apache.spark.sql.functions._
val hour_diff = udf((fromTimestamp: Timestamp, toTimestamp: Timestamp) => {
  scala.util.Try((toTimestamp.getTime - fromTimestamp.getTime) / 3600000.0).getOrElse(0.0)
})

// COMMAND ----------

// DBTITLE 0,Enrich trips with origin and destination ports
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame

case class Port(id: Long, name: String)

// A specific location may match multiple polygons from different ports
// we find the port with the most overlapping polygons (maximizing Jaccard coefficient)
val getPortCandidate = udf((xs: Seq[Row]) => {
  val b = xs.maxBy(_.getAs[Long]("count"))
  Port(b.getAs[Long]("portId"), b.getAs[String]("portName"))
})

// We ensure to use same granularity as per catchment area so that we can join both datasets on H3
def getPort(df: DataFrame): DataFrame = {
  df
    .select(col("tripId"), toH3(col("latitude"), col("longitude"), lit(5)).as("h3"))
    .join(portIndexDf, List("h3"))
    .groupBy("tripId", "portId", "portName")
    .count()
    .withColumn("port", struct(col("portId"), col("portName"), col("count")))
    .groupBy("tripId")
    .agg(collect_list(col("port")).as("candidates"))
    .select(col("tripId"), getPortCandidate(col("candidates")).as("port"))  
}

// Run process twice, once for originating port, once for destination port 
// We only keep inner join, hence ignoring trips originating / at destination to outside of the US
val orgPortDf = getPort(trips.select("tripId", "orgGeo.latitude", "orgGeo.longitude")).withColumnRenamed("port", "orgPort")
val dstPortDf = getPort(trips.select("tripId", "dstGeo.latitude", "dstGeo.longitude")).withColumnRenamed("port", "dstPort")
val portDf = orgPortDf.join(dstPortDf, List("tripId"))

trips
  .join(portDf, List("tripId"))
  .select(
    col("tripId"),
    col("callSign"),
    col("mmsi"),
    col("vesselName"),
    col("flag"),
    hour_diff(col("orgTime"), col("dstTime")).as("duration"),
    col("distance"),
    col("orgTime"),
    col("orgGeo"),
    col("orgPort.id").as("orgPortId"),
    col("orgPort.name").as("orgPortName"),
    col("dstTime"),
    col("dstGeo"),
    col("dstPort.id").as("dstPortId"),
    col("dstPort.name").as("dstPortName")
  )
  .filter(col("orgPortId") =!= col("dstPortId"))
  .filter(col("distance") > 0) // must have sailed at least one nautic mile
  .filter(col("duration") > 5) // must have been sailing for at least 5 hours
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable("esg.cargos_trips")

// COMMAND ----------

// DBTITLE 0,Optimize delta table
// MAGIC %sql
// MAGIC OPTIMIZE esg.cargos_trips ZORDER BY (tripId, orgPortId, dstPortId)

// COMMAND ----------

// DBTITLE 1,We only have 15,000 well defined port to port trips (US coastal data only)
// MAGIC %sql
// MAGIC SELECT * FROM esg.cargos_trips
// MAGIC ORDER BY duration DESC

// COMMAND ----------

// MAGIC %md
// MAGIC As represented below using [KeplerGL](https://kepler.gl/) visualisation (an extra step is required to render HTML in Databricks notebook), this apparent complex approach helped us drammatically summarize billions of data points into well defined trips with a clear ports of origin / destinations, resulting into a much more "digestible" heatmap visualisation. We can even observe traffic alongside the Mississipi river.

// COMMAND ----------

// DBTITLE 0,Extra step to render KeplerGl vis.
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

// DBTITLE 0,Kepler config
// MAGIC %python
// MAGIC config = {
// MAGIC   "version": "v1",
// MAGIC   "config": {
// MAGIC     "visState": {
// MAGIC       "filters": [],
// MAGIC       "layers": [
// MAGIC         {
// MAGIC           "id": "6hsbp6",
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
// MAGIC       "latitude": 48.72985668568491,
// MAGIC       "longitude": -112.87184367328193,
// MAGIC       "pitch": 0,
// MAGIC       "zoom": 2.7801407685529185,
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

// DBTITLE 1,"De-noising" maritime traffic
// MAGIC %python
// MAGIC from pyspark.sql import functions as F
// MAGIC from keplergl import KeplerGl 
// MAGIC 
// MAGIC trips = spark.read.table("esg.cargos_trips")
// MAGIC points = spark.read.table("esg.cargos_points")
// MAGIC 
// MAGIC df = trips \
// MAGIC   .join(points, ["tripId"]) \
// MAGIC   .withColumn("h3", F.expr("h3(latitude, longitude, 6)")) \
// MAGIC   .groupBy("h3").count().toPandas()
// MAGIC 
// MAGIC KeplerGl(height=800, data={'data': df}, config=config)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Extracting flags of convenience
// MAGIC *[...] With the growing concern to move towards a cleaner shipping industry, shipbreaking will play an increasingly important role in corporate activities and investor decisions. Current regulatory frameworks looking to govern shipbreaking activities include the Basel Convention, OECD and ILO guidelines, and the European Union Ship Recycling Regulation, which came into force in January 2019. However, many boats sail under flags of convenience, including Panama, Liberia and Marshall Islands, making it possible to escape the rules laid down by international organizations and governments. [source](https://www.sustainalytics.com/esg-blog/shipbreaking-clean-shipping-in-deep-water/)*. 

// COMMAND ----------

// DBTITLE 1,No flags of convenience for vessels operating in US only
// MAGIC %sql
// MAGIC SELECT 
// MAGIC   callSign, 
// MAGIC   vesselName, 
// MAGIC   COLLECT_SET(flag) as flags 
// MAGIC FROM esg.cargos_trips 
// MAGIC WHERE flag is not null 
// MAGIC GROUP BY callSign, vesselName 
// MAGIC HAVING SIZE(flags) > 1

// COMMAND ----------

// DBTITLE 1,2 international ships operated under flag of convenience
// MAGIC %sql
// MAGIC CREATE OR REPLACE TEMPORARY VIEW convenience_ships AS
// MAGIC SELECT 
// MAGIC   callSign, 
// MAGIC   vesselName, 
// MAGIC   COLLECT_SET(flag(mmsi)) as flags 
// MAGIC FROM esg.cargos_points
// MAGIC WHERE flag(mmsi) is not null 
// MAGIC GROUP BY callSign, vesselName 
// MAGIC HAVING SIZE(flags) > 1;
// MAGIC 
// MAGIC SELECT * FROM convenience_ships

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Flags of convenience operating in the US
// MAGIC 
// MAGIC PEAK PEGASUS (IMO: 9634830) is a Bulk Carrier that was built in 2013 (7 years ago) and is sailing under the flag of Liberia
// MAGIC <br>
// MAGIC <img src="https://photos.marinetraffic.com/ais/showphoto.aspx?shipid=410512" width=300>
// MAGIC 
// MAGIC GLORIA ELENA (IMO: 8012566) is a Cement Carrier that was built in 1981 (39 years ago) and is sailing under the flag of Mexico
// MAGIC <br>
// MAGIC <img src="https://photos.marinetraffic.com/ais/showphoto.aspx?shipid=403769" width=300>

// COMMAND ----------

// DBTITLE 0,Kepler configuration
// MAGIC %python
// MAGIC config={
// MAGIC   "version": "v1",
// MAGIC   "config": {
// MAGIC     "visState": {
// MAGIC       "filters": [],
// MAGIC       "layers": [
// MAGIC         {
// MAGIC           "id": "nia1xib",
// MAGIC           "type": "point",
// MAGIC           "config": {
// MAGIC             "dataId": "data",
// MAGIC             "label": "Point",
// MAGIC             "color": [
// MAGIC               18,
// MAGIC               147,
// MAGIC               154
// MAGIC             ],
// MAGIC             "columns": {
// MAGIC               "lat": "latitude",
// MAGIC               "lng": "longitude",
// MAGIC               "altitude": None
// MAGIC             },
// MAGIC             "isVisible": True,
// MAGIC             "visConfig": {
// MAGIC               "radius": 16.5,
// MAGIC               "fixedRadius": False,
// MAGIC               "opacity": 0.12,
// MAGIC               "outline": False,
// MAGIC               "thickness": 2,
// MAGIC               "strokeColor": None,
// MAGIC               "colorRange": {
// MAGIC                 "name": "Custom Palette",
// MAGIC                 "type": "custom",
// MAGIC                 "category": "Custom",
// MAGIC                 "colors": [
// MAGIC                   "E8D61F",
// MAGIC                   "B02485"
// MAGIC                 ]
// MAGIC               },
// MAGIC               "strokeColorRange": {
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
// MAGIC               "radiusRange": [
// MAGIC                 0,
// MAGIC                 50
// MAGIC               ],
// MAGIC               "filled": True
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
// MAGIC               "name": "vesselName",
// MAGIC               "type": "string"
// MAGIC             },
// MAGIC             "colorScale": "ordinal",
// MAGIC             "strokeColorField": None,
// MAGIC             "strokeColorScale": "quantile",
// MAGIC             "sizeField": None,
// MAGIC             "sizeScale": "linear"
// MAGIC           }
// MAGIC         }
// MAGIC       ],
// MAGIC       "interactionConfig": {
// MAGIC         "tooltip": {
// MAGIC           "fieldsToShow": {
// MAGIC             "data": [
// MAGIC               {
// MAGIC                 "name": "vesselName",
// MAGIC                 "format": None
// MAGIC               },
// MAGIC               {
// MAGIC                 "name": "flag",
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
// MAGIC       "latitude": 25.211803691896275,
// MAGIC       "longitude": -89.32723176887419,
// MAGIC       "pitch": 0,
// MAGIC       "zoom": 4.479181521626277,
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

// DBTITLE 1,Where do these ships operate? 
// MAGIC %python
// MAGIC 
// MAGIC from pyspark.sql import functions as F
// MAGIC from keplergl import KeplerGl 
// MAGIC 
// MAGIC convenience_ships = spark.read.table("convenience_ships").toPandas()['callSign']
// MAGIC points = spark.read.table("esg.cargos_points") \
// MAGIC   .filter(F.col("callSign").isin(list(convenience_ships))) \
// MAGIC   .withColumn("flag", F.expr("flag(mmsi)")) \
// MAGIC   .select("vesselName", "flag", "latitude", "longitude").toPandas()
// MAGIC 
// MAGIC KeplerGl(height=500, data={'data': points}, config=config)

// COMMAND ----------

// MAGIC %md
// MAGIC As reported in above picture, these 2 ships have been operating in the gulf of Mexico in 2018 under a Liberia or Gabon flag (Peak Pegasus in purple). There are a variety of factors for sailing under flag of convenience, but least disciplined ship owners tend to register vessels in countries that impose less regulations. Consequently, ships bearing a flag of convenience are often characterized by their poor conditions, inadequately trained crews, and frequent collisions and cause a serious environmental and safety concern that are worth being identified. Another pattern that is worth mentioning is the apparent lack of data captured by AIS (on purpose or not), making this risk even harder to identify and quantify.

// COMMAND ----------

// MAGIC %md
// MAGIC ---
// MAGIC + <a href="$./00_vessel_context">STAGE0</a>: Home page
// MAGIC + <a href="$./01_vessel_etl">STAGE1</a>: Download vessel AIS tracking data
// MAGIC + <a href="$./02_vessel_trips">STAGE2</a>: Session data points into trips
// MAGIC + <a href="$./03_vessel_markov">STAGE3</a>: Journey optimization
// MAGIC + <a href="$./04_vessel_predict">STAGE4</a>: Predicting vessel destination
// MAGIC ---
