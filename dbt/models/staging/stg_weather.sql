-- models/staging/stg_weather.sql
-- Silver layer: clean and cast the raw weather data
-- Fixes float precision and casts date string to DATE type
-- No aggregation, no business logic — just clean raw data

{{ config(materialized='table') }}

SELECT
    SAFE_CAST(date AS DATE)                    AS date,
    city,
    ROUND(CAST(temperature_max AS FLOAT64), 1) AS temperature_max,
    ROUND(CAST(temperature_min AS FLOAT64), 1) AS temperature_min,
    ROUND(CAST(precipitation_sum AS FLOAT64), 1) AS precipitation_sum,
    ROUND(CAST(wind_speed_max AS FLOAT64), 1)  AS wind_speed_max,
    CAST(weather_code AS INT64)                AS weather_code

FROM {{ source('weather_raw', 'weather_raw_los_angeles') }}

WHERE date IS NOT NULL
  AND temperature_max IS NOT NULL
  AND temperature_min IS NOT NULL