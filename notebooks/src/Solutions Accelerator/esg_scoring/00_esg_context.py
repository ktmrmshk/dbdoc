# Databricks notebook source
# MAGIC %md
# MAGIC <img src=files/antoine.amend@databricks.com/databricks_fsi_white.png width=600px>

# COMMAND ----------

# MAGIC %md
# MAGIC # ESG - context
# MAGIC 
# MAGIC The future of finance goes hand in hand with social responsibility, environmental stewardship and corporate ethics. In order to stay competitive, Financial Services Institutions (FSI)  are increasingly  disclosing more information about their **environmental, social and governance** (ESG) performance. By better understanding and quantifying the sustainability and societal impact of any investment in a company or business, FSIs can mitigate reputation risk and maintain the trust with both their clients and shareholders. At Databricks, we increasingly hear from our customers that ESG has become a C-suite priority. This is not solely driven by altruism but also by economics: [Higher ESG ratings are generally positively correlated with valuation and profitability while negatively correlated with volatility](https://corpgov.law.harvard.edu/2020/01/14/esg-matters/). In this demo, we offer a novel approach to sustainable finance by combining NLP techniques and graph analytics to extract key strategic ESG initiatives and learn companies' relationships in a global market.
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./00_esg_context">STAGE0</a>: Home page
# MAGIC + <a href="$./01_esg_report">STAGE1</a>: Using NLP to extract key ESG initiatives PDF reports
# MAGIC + <a href="$./02_esg_scoring">STAGE2</a>: Introducing a novel approach to ESG scoring using graph analytics
# MAGIC + <a href="$./03_esg_market">STAGE3</a>: Applying ESG to market risk calculations
# MAGIC + <a href="$./04_esg_dashboard">STAGE4</a>: Package all visualizations into powerful dashboards
# MAGIC ---
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

displayHTML("""<iframe src="https://www.youtube.com/embed/I9QO4Dpkb7c?&mute=1"></iframe>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Context
# MAGIC 
# MAGIC Using **NLP**, news articles and **graph analytics**, we will demonstrate how asset managers can assess the sustainability of their investments and empower their business with a holistic and data driven view to their environmental, social and governance strategies. Specifically, we will **extract the key ESG initiatives** as communicated in yearly PDF reports and compare these with the actual media coverage from news analytics data. In the second part of this demo, we will learn the connections between companies and understand the **positive or negative ESG consequences** these connections may have to your business. Finally, we will evaluate the impact ESG has to your investment strategy and **market risk**.
# MAGIC 
# MAGIC 
# MAGIC <img src="/files/shared_uploads/finserv/images/esg_workflow.png" alt="logical_flow" width="800">
