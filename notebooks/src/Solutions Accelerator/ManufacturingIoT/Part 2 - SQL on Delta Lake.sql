-- Databricks notebook source
-- MAGIC %md # 製造におけるIoT分析 on Databricks
-- MAGIC ## Part 2: SQL on the Manufacturing Delta Lake
-- MAGIC 
-- MAGIC Databricks [SQL Analytics](https://databricks.com/product/sql-analytics)は、アナリストに特化した直感的なインターフェイスを使用して、Data Lake Deltaのテーブルを照会するために使用できます。また、PowerBIのようなBIツールは、SQLエンドポイントに接続することで、データサイロにデータを移すことなく、製造データに対して高速でスケーラブルかつセキュアなアドホックBIを実現します。
-- MAGIC 
-- MAGIC <img src="https://sguptasa.blob.core.windows.net/random/iiot_blog/sql_dashboard_manu.gif" width=800>
-- MAGIC 
-- MAGIC SQL Analyicsはプライベート・プレビューなので、このノートブックにはクエリが含まれています。上のダッシュボードを再現するには、以下の手順で行います。
-- MAGIC 1. SQLエンドポイントの作成
-- MAGIC 2. データソース設定にADLSにアクセスするためのサービスプリンシパルを追加する。
-- MAGIC 3. 以下のクエリをSQL Analyticsの「クエリ」にコピーして実行する。
-- MAGIC 4. facilities」と「date」の範囲でパラメータを作成します。
-- MAGIC 5. クエリからカウンター、時系列チャート、マップ、その他の視覚化を構築する
-- MAGIC 6. 作成したビジュアライゼーションを使用してダッシュボードを組み立てる
-- MAGIC 
-- MAGIC **注意**: このクエリは、SQL Analyticsで使用するためにパラメータ化（例：`{{ facilities }}`）されているため、ノートブックではそのままでは実行されません。

-- COMMAND ----------

-- MAGIC %md ### 在庫分析

-- COMMAND ----------

select distinct i.facilityid, 
    last(inventory) over (partition by i.facilityid order by date) as inventory,
    f.state
from manufacturing.parts_inventory i join manufacturing.facilities f
on (i.facilityid = f.facilityid)
where i.facilityid in ({{ facilities }})

-- COMMAND ----------

-- MAGIC %md ### エンリッチド分析

-- COMMAND ----------

select * from manufacturing.sensors_enriched
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ### 生データの時系列プロット

-- COMMAND ----------

select * from manufacturing.sensors_raw
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ###　読み込みデータの集計

-- COMMAND ----------

select count(*) as Readings, 
    avg(temperature) as Temperature,
    avg(humidity) as Humidity,
    avg(pressure) as Pressure,
    avg(moisture) as Moisture,
    avg(oxygen) as Oxygen,
    avg(radiation) as Radiation,
    avg(conductivity) as Conductivity
from manufacturing.sensors_raw
where facilityid in ({{ facilityids }})
    and date between '{{ date.start }}' and '{{ date.end }}' 

-- COMMAND ----------

-- MAGIC %md ### 設備一覧
-- MAGIC このリストを使って、他のクエリに対するクエリベースのフィルタを作成することができます。

-- COMMAND ----------

select facilityid from manufacturing.facilities

-- COMMAND ----------

-- MAGIC %md ### 施設マップ
-- MAGIC このクエリを使用して、施設の容量と位置をマッピングしたジオマップを作成することができます。

-- COMMAND ----------

select * from manufacturing.facilities where facilityid in ({{ facilityids }})
