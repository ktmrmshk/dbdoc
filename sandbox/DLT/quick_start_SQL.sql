-- Databricks notebook source
--%fs ls /databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/2015_2_clickstream.json

-- COMMAND ----------

-- create or replace temp view ktmr_clickstream_raw
-- comment "This is a commnet!!!!"
-- as select * from json.`/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/2015_2_clickstream.json`;

-- select * from ktmr_clickstream_raw

-- COMMAND ----------

-- bronze
create live table ktmr_clickstream_raw
comment "This is a commnet!!!!"
as select * from json.`/databricks-datasets/wikipedia-datasets/data-001/clickstream/raw-uncompressed-json/2015_2_clickstream.json`

-- COMMAND ----------

-- silver
create live table ktmr_clickstream_clean (
  constraint valid_current_page expect (current_page_title is not null),
  constraint valid_count expect ( click_count > 0) on violation fail update
)
comment "this is silver table!"
as select
  curr_title AS current_page_title,
  CAST(n AS INT) AS click_count,
  prev_title AS previous_page_title
from live.ktmr_clickstream_raw

-- COMMAND ----------

-- gold
CREATE LIVE TABLE top_spark_referers
COMMENT "A table containing the top pages linking to the Apache Spark page."
AS SELECT
  previous_page_title as referrer,
  click_count
FROM live.ktmr_clickstream_clean
WHERE current_page_title = 'Apache_Spark'
ORDER BY click_count DESC
LIMIT 10

-- COMMAND ----------


