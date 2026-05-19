-- models/marts/mart_weather_daily.sql
-- Gold layer: analytics-ready weather features for Los Angeles
-- This is what the forecasting model consumes next weekend

{{ config(materialized='table') }}

WITH base AS (
    SELECT
        date,
        city,
        temperature_max,
        temperature_min,
        precipitation_sum,
        wind_speed_max,
        weather_code,

        ROUND(temperature_max - temperature_min, 1) AS temperature_range,

        ROUND(
            AVG(temperature_max) OVER (
                PARTITION BY city
                ORDER BY date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 1
        ) AS rolling_avg_temp_7d,

        LAG(temperature_max, 3) OVER (
            PARTITION BY city ORDER BY date
        ) AS temp_lag_3,

        LAG(temperature_max, 7) OVER (
            PARTITION BY city ORDER BY date
        ) AS temp_lag_7,

        AVG(temperature_max) OVER (
            PARTITION BY city, EXTRACT(MONTH FROM date)
        ) AS monthly_avg_temp,

        STDDEV(temperature_max) OVER (
            PARTITION BY city, EXTRACT(MONTH FROM date)
        ) AS monthly_stddev_temp,

        CASE
            WHEN EXTRACT(MONTH FROM date) IN (12, 1, 2)  THEN 'Winter'
            WHEN EXTRACT(MONTH FROM date) IN (3, 4, 5)   THEN 'Spring'
            WHEN EXTRACT(MONTH FROM date) IN (6, 7, 8)   THEN 'Summer'
            WHEN EXTRACT(MONTH FROM date) IN (9, 10, 11) THEN 'Fall'
        END AS season,

        CASE WHEN temperature_max > 35 THEN 1 ELSE 0 END AS is_extreme_heat,
        CASE WHEN precipitation_sum > 10 THEN 1 ELSE 0 END AS is_heavy_rain,

        CASE weather_code
            WHEN 0  THEN 'Clear Sky'
            WHEN 1  THEN 'Mainly Clear'
            WHEN 2  THEN 'Partly Cloudy'
            WHEN 3  THEN 'Overcast'
            WHEN 45 THEN 'Fog'
            WHEN 48 THEN 'Icy Fog'
            WHEN 51 THEN 'Light Drizzle'
            WHEN 53 THEN 'Moderate Drizzle'
            WHEN 55 THEN 'Dense Drizzle'
            WHEN 61 THEN 'Slight Rain'
            WHEN 63 THEN 'Moderate Rain'
            WHEN 65 THEN 'Heavy Rain'
            WHEN 71 THEN 'Slight Snow'
            WHEN 73 THEN 'Moderate Snow'
            WHEN 75 THEN 'Heavy Snow'
            WHEN 80 THEN 'Slight Showers'
            WHEN 81 THEN 'Moderate Showers'
            WHEN 82 THEN 'Violent Showers'
            WHEN 95 THEN 'Thunderstorm'
            ELSE 'Unknown'
        END AS weather_description

    FROM {{ ref('stg_weather') }}
),

final AS (
    SELECT
        date,
        city,
        temperature_max,
        temperature_min,
        precipitation_sum,
        wind_speed_max,
        weather_code,
        temperature_range,
        rolling_avg_temp_7d,
        temp_lag_3,
        temp_lag_7,
        season,
        is_extreme_heat,
        is_heavy_rain,
        CASE
            WHEN monthly_stddev_temp IS NOT NULL
            AND ABS(temperature_max - monthly_avg_temp) > 2 * monthly_stddev_temp
            THEN 1
            ELSE 0
        END AS anomaly_flag

    FROM base
)

SELECT * FROM final
ORDER BY date ASC