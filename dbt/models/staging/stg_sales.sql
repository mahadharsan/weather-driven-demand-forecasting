-- models/staging/stg_sales.sql
-- Silver layer: clean and cast raw CA_3 FOODS daily sales data
-- No joining, no aggregation — just clean types

{{ config(materialized='table') }}

SELECT
    SAFE_CAST(date AS DATE)        AS date,
    CAST(units_sold AS INT64)      AS units_sold,
    CAST(snap_ca AS INT64)         AS snap_ca,
    CAST(is_event AS INT64)        AS is_event,
    CAST(day_of_week AS INT64)     AS day_of_week

FROM {{ source('sales_raw', 'sales_raw_ca3_foods') }}

WHERE date IS NOT NULL
  AND units_sold IS NOT NULL