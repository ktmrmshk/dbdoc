-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC # Simplify ETL with Delta Live Table
-- MAGIC 
-- MAGIC DLT makes Data Engineering accessible for all. Just declare your transformations in SQL or Pythin, and DLT will handle the Data Engineering complexity for you.
-- MAGIC 
-- MAGIC <img style="float:right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-1.png" width="700"/>
-- MAGIC 
-- MAGIC **Accelerate ETL development** <br/>
-- MAGIC Enable analysts and data engineers to innovate rapidly with simple pipeline development and maintenance 
-- MAGIC 
-- MAGIC **Remove operational complexity** <br/>
-- MAGIC By automating complex administrative tasks and gaining broader visibility into pipeline operations
-- MAGIC 
-- MAGIC **Trust your data** <br/>
-- MAGIC With built-in quality controls and quality monitoring to ensure accurate and useful BI, Data Science, and ML 
-- MAGIC 
-- MAGIC **Simplify batch and streaming** <br/>
-- MAGIC With self-optimization and auto-scaling data pipelines for batch or streaming processing 
-- MAGIC 
-- MAGIC ## Our Delta Live Table pipeline
-- MAGIC 
-- MAGIC We'll be using as input a raw dataset containing information on our customers Loan and historical transactions. 
-- MAGIC 
-- MAGIC Our goal is to ingest this data in near real time and build table for our Analyst team while ensuring data quality.
-- MAGIC 
-- MAGIC <!-- do not remove -->
-- MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fdlt%2Fnotebook_dlt_sql&dt=DLT">
-- MAGIC <!-- [metadata={"description":"Full DLT demo, going into details. Use loan dataset",
-- MAGIC  "authors":["dillon.bostwick@databricks.com"],
-- MAGIC  "db_resources":{},
-- MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["autoloader", "dlt"]}}] -->

-- COMMAND ----------

-- MAGIC %md-sandbox 
-- MAGIC 
-- MAGIC ## Bronze layer: ingestion incremental data leveraging Databricks Autoloader
-- MAGIC 
-- MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-2.png" width="500"/>
-- MAGIC 
-- MAGIC Our raw data is being sent to a blob storage. 
-- MAGIC 
-- MAGIC Autoloader simplify this ingestion, including schema inference, schema evolution while being able to scale to millions of incoming files. 
-- MAGIC 
-- MAGIC Autoloader is available in SQL using the `cloud_files` function and can be used with a variety of format (json, csv, avro...):
-- MAGIC 
-- MAGIC 
-- MAGIC #### INCREMENTAL LIVE TABLE 
-- MAGIC Defining tables as `INCREMENTAL` will garantee that you only consume new incoming data. Without incremental, you will scan and ingest all the data available at once. See the [documentation](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-incremental-data.html) for more details

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE BZ_raw_txs
  COMMENT "New raw loan data incrementally ingested from cloud object storage landing zone"
  TBLPROPERTIES ("quality" = "bronze")
AS SELECT * FROM cloud_files('/demo/dlt_loan/landing', 'json')

-- COMMAND ----------

CREATE LIVE TABLE ref_accounting_treatment
  COMMENT "Lookup mapping for accounting codes"
AS SELECT * FROM delta.`/demo/dlt_loan/ref_accounting_treatment/`

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE BZ_reference_loan_stats
  COMMENT "Raw historical transactions"
  TBLPROPERTIES ("quality" = "bronze")
AS SELECT * FROM cloud_files('/databricks-datasets/lending-club-loan-stats/LoanStats_*', 'csv')

-- COMMAND ----------

-- MAGIC %md-sandbox 
-- MAGIC 
-- MAGIC ## Silver layer: joining tables while ensuring data quality
-- MAGIC 
-- MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-3.png" width="500"/>
-- MAGIC 
-- MAGIC Once the bronze layer is defined, we'll create the sliver layers by Joining data. Note that bronze tables are referenced using the `LIVE` spacename. 
-- MAGIC 
-- MAGIC To consume only increment from the Bronze layer like `BZ_raw_txs`, we'll be using the `stream` keyworkd: `stream(LIVE.BZ_raw_txs)`
-- MAGIC 
-- MAGIC Note that we don't have to worry about compactions, DLT handles that for us.
-- MAGIC 
-- MAGIC #### Expectations
-- MAGIC By defining expectations (`CONSTRANT <name> EXPECT <condition>`), you can enforce and track your data quality. See the [documentation](https://docs.databricks.com/data-engineering/delta-live-tables/delta-live-tables-expectations.html) for more details

-- COMMAND ----------

CREATE INCREMENTAL LIVE TABLE SV_cleaned_new_txs (
  CONSTRAINT `Payments should be this year`  EXPECT (next_payment_date > date('2020-12-31')),
  CONSTRAINT `Balance should be positive`    EXPECT (balance > 0 AND arrears_balance > 0) ON VIOLATION DROP ROW,
  CONSTRAINT `Cost center must be specified` EXPECT (cost_center_code IS NOT NULL) ON VIOLATION FAIL UPDATE
  -- Roadmap: Quarantine
)
  COMMENT "Livestream of new transactions, cleaned and compliant"
  TBLPROPERTIES ("quality" = "silver")
AS SELECT txs.*, rat.id as accounting_treatment FROM stream(LIVE.BZ_raw_txs) txs
  INNER JOIN live.ref_accounting_treatment rat ON txs.accounting_treatment_id = rat.id

-- COMMAND ----------

CREATE LIVE TABLE SV_historical_txs
  COMMENT "Historical loan transactions"
  TBLPROPERTIES ("quality" = "silver")
AS SELECT a.* FROM LIVE.BZ_reference_loan_stats a
  INNER JOIN LIVE.ref_accounting_treatment b USING (id)

-- COMMAND ----------

-- MAGIC %md-sandbox 
-- MAGIC 
-- MAGIC ## Gold layer
-- MAGIC 
-- MAGIC <img style="float: right; padding-left: 10px" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/dlt-golden-demo-4.png" width="500"/>
-- MAGIC 
-- MAGIC Our last step is to materialize the Gold Layer.
-- MAGIC 
-- MAGIC Because these tables will be requested at scale using a SQL Endpoint, we'll add Zorder at the table level to ensure faster queries using `pipelines.autoOptimize.zOrderCols`, and DLT will handle the rest.

-- COMMAND ----------

CREATE LIVE TABLE GL_total_loan_balances_1
  COMMENT "Combines historical and new loan data for unified rollup of loan balances"
  TBLPROPERTIES (
    "quality" = "gold",
    "pipelines.autoOptimize.zOrderCols" = "location_code"
  )
AS SELECT sum(revol_bal)  AS bal, addr_state   AS location_code FROM live.SV_historical_txs  GROUP BY addr_state
  UNION SELECT sum(balance) AS bal, country_code AS location_code FROM live.SV_cleaned_new_txs GROUP BY country_code

-- COMMAND ----------

CREATE LIVE TABLE GL_total_loan_balances_2
  COMMENT "Combines historical and new loan data for unified rollup of loan balances"
  TBLPROPERTIES ("quality" = "gold")
AS SELECT sum(revol_bal)  AS bal, addr_state   AS location_code FROM live.SV_historical_txs  GROUP BY addr_state
  UNION SELECT sum(balance) AS bal, country_code AS location_code FROM live.SV_cleaned_new_txs GROUP BY country_code

-- COMMAND ----------

CREATE LIVE VIEW GL_new_loan_balances_by_cost_center
  COMMENT "Live view of new loan balances for consumption by different cost centers"
  TBLPROPERTIES (
    "quality" = "gold",
    "pipelines.autoOptimize.zOrderCols" = "cost_center_code"
  )
AS SELECT sum(balance), cost_center_code FROM live.SV_cleaned_new_txs
  GROUP BY cost_center_code

-- COMMAND ----------

CREATE LIVE VIEW GL_new_loan_balances_by_country
  COMMENT "Live view of new loan balances per country"
  TBLPROPERTIES (
    "quality" = "gold",
    "pipelines.autoOptimize.zOrderCols" = "country_code"
  )
AS SELECT sum(count), country_code FROM live.SV_cleaned_new_txs GROUP BY country_code

-- COMMAND ----------

-- MAGIC %md ## Next steps
-- MAGIC 
-- MAGIC Your DLT pipeline is ready to be started.
-- MAGIC 
-- MAGIC Open the DLT menu, create a pipeline and select this notebook to run it. To generate sample data, please run the [companion notebook]($./00-Loan-Data-Generator) (make sure the path where you read and write the data are the same!)
-- MAGIC 
-- MAGIC Datas Analyst can start using DBSQL to analyze data and track our Loan metrics.  Data Scientist can also access the data to start building models to predict payment default or other more advanced use-cases.

-- COMMAND ----------

-- MAGIC %md ## Tracking data quality
-- MAGIC 
-- MAGIC Expectations stats are automatically available as system table.
-- MAGIC 
-- MAGIC This information let you monitor your data ingestion quality. 
-- MAGIC 
-- MAGIC You can leverage DBSQL to request these table and build custom alerts based on the metrics your business is tracking.
-- MAGIC 
-- MAGIC 
-- MAGIC See [how to access your DLT metrics]($./02-Log-Analysis)
-- MAGIC 
-- MAGIC <img width="500" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/retail-dlt-data-quality-dashboard.png">
-- MAGIC 
-- MAGIC <a href="https://e2-demo-field-eng.cloud.databricks.com/sql/dashboards/6f73dd1b-17b1-49d0-9a11-b3772a2c3357-dlt---retail-data-quality-stats?o=1444828305810485" target="_blank">Data Quality Dashboard example</a>
