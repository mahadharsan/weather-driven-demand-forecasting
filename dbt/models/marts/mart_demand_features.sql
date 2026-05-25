-- models/marts/mart_demand_features.sql
-- Gold layer: joins weather and sales data 
-- Creates lag and rolling average features for demand forecasting
-- This is the single table the forecasting model trains on

{{ config(materialized='table') }}

WITH sales AS (
    SELECT
        date,
        units_sold,
        snap_ca,
        is_event,
        day_of_week
    FROM {{ ref('stg_sales') }}
),

weather AS (
    SELECT
        date,
        temperature_max,
        temperature_min,
        precipitation_sum,
        anomaly_flag,
        rolling_avg_temp_7d,
        is_extreme_heat,
        is_heavy_rain,
        weather_code,
        weather_description,
        season,
        temp_lag_3,
        temp_lag_7
    FROM {{ ref('mart_weather_daily') }}
),

joined AS (
    SELECT
        s.date,
        s.units_sold,
        s.snap_ca,
        s.is_event,
        s.day_of_week,
        w.temperature_max,
        w.temperature_min,
        w.precipitation_sum,
        w.anomaly_flag,
        w.rolling_avg_temp_7d,
        w.is_extreme_heat,
        w.is_heavy_rain,
        w.weather_code,
        w.weather_description,
        w.season,
        w.temp_lag_3,
        w.temp_lag_7
    FROM sales s
    LEFT JOIN weather w ON s.date = w.date
),

final AS (
    SELECT
        date,
        units_sold,
        snap_ca,
        is_event,
        day_of_week,
        temperature_max,
        temperature_min,
        precipitation_sum,
        anomaly_flag,
        rolling_avg_temp_7d,
        is_extreme_heat,
        is_heavy_rain,
        weather_code,
        weather_description,
        season,
        temp_lag_3,
        temp_lag_7,

        -- Sales lag features
        LAG(units_sold, 7) OVER (
            ORDER BY date
        ) AS sales_lag_7,

        LAG(units_sold, 14) OVER (
            ORDER BY date
        ) AS sales_lag_14,

        -- Rolling average sales
        ROUND(
            AVG(units_sold) OVER (
                ORDER BY date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 1
        ) AS sales_rolling_avg_7d,

        ROUND(
            AVG(units_sold) OVER (
                ORDER BY date
                ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
            ), 1
        ) AS sales_rolling_avg_14d

    FROM joined
)

SELECT * FROM final
ORDER BY date ASC