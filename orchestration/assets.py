# orchestration/assets.py
# Wraps existing ingest and load scripts as Dagster assets
# Each @asset represents one piece of data this pipeline produces

import sys
from pathlib import Path
from dagster import asset, Output, MetadataValue
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from ingestion.ingest import fetch_weather_data, validate_and_convert, upload_to_gcs
from loading.load_to_bq import load_parquet_to_bigquery


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asset(
    group_name="weather_pipeline",
    description="Fetches LA weather from Open-Meteo API and uploads Parquet to GCS bronze layer"
)
def raw_weather_parquet():
    config = load_config()
    df     = fetch_weather_data(config)
    table  = validate_and_convert(df)
    upload_to_gcs(table, config)

    return Output(
        value=True,
        metadata={
            "rows":       MetadataValue.int(len(df)),
            "city":       MetadataValue.text(config["api"]["city"]),
            "date_range": MetadataValue.text(
                f"{config['api']['start_date']} to {config['api']['end_date']}"
            )
        }
    )


@asset(
    deps=[raw_weather_parquet],
    group_name="weather_pipeline",
    description="Loads Parquet from GCS into BigQuery raw table"
)
def raw_weather_bq_table():
    config = load_config()
    load_parquet_to_bigquery(config)

    return Output(
        value=True,
        metadata={
            "table": MetadataValue.text(
                f"{config['gcp']['project_id']}."
                f"{config['gcp']['dataset_id']}."
                f"{config['pipeline']['raw_table']}"
            )
        }
    )