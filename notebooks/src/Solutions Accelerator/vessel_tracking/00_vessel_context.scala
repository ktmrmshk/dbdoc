// Databricks notebook source
// MAGIC %md
// MAGIC # Vessel Tracking - context
// MAGIC 
// MAGIC *The benefits of Environmental, Social and Governance (ESG) is well understood across the financial service industry, but the benefits of ESG goes beyond sustainable investments. What recent experience has taught us is that high social values and good governance emerged as key indicators of resilience throughout the COVID-19 pandemic. Large retailers that already use ESG to monitor the performance of their supply chain have been able to leverage this information to better navigate the challenges of global lockdowns, ensuring a constant flow of goods and products to communities. As reported in an [article](https://corpgov.law.harvard.edu/2020/08/14/the-other-s-in-esg-building-a-sustainable-and-resilient-supply-chain/) from Harvard Law School Forum on Corporate Governance, [...] companies that invest in [ESG] also benefit from competitive advantages, faster recovery from disruptions.*
// MAGIC 
// MAGIC <br>
// MAGIC *“High-quality businesses that adhere to sound ESG practices will outperform those that do not.”*
// MAGIC <br>
// MAGIC Tidjane Thiam - Former CEO of Credit Suisse
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
// MAGIC In this solution accelerator, we demonstrate a novel approach to Supply Chain analytics by combining geospatial techniques and predictive analytics for logistics companies not only to reduce their carbon footprint, improve working conditions and enhance regulatory compliance but also to use that information to adapt to emerging threats, in real time. 
// MAGIC 
// MAGIC <img src="/files/antoine.amend/images/esg2_workflow.png" alt="logical_flow" width="800">

// COMMAND ----------


