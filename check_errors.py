import os
import yaml
import pandas as pd
from google.cloud import bigquery

config = yaml.safe_load(open('config/settings.yaml'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['gcp']['credentials_path']
client = bigquery.Client(project=config['gcp']['project_id'])

query = """
    SELECT 
        f.date,
        f.actual_units_sold,
        f.xgb_no_weather,
        ABS(f.actual_units_sold - f.xgb_no_weather) as error,
        w.temperature_max,
        w.precipitation_sum,
        w.anomaly_flag,
        w.is_extreme_heat,
        w.is_heavy_rain,
        w.weather_description
    FROM `climate-de-pipeline.weather_marts.mart_forecast_output` f
    LEFT JOIN `climate-de-pipeline.weather_marts.mart_weather_daily` w
        ON CAST(f.date AS DATE) = w.date
    ORDER BY error DESC
    LIMIT 20
"""

df = client.query(query).to_dataframe()
print(df.to_string())