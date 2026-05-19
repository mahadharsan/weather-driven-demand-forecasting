# ingestion/ingest.py
# Sole responsibility: fetch weather data from Open-Meteo API
# and upload it as a Parquet file to GCS bronze layer
# Does NOT transform, does NOT load to BigQuery

import os
import sys
import yaml
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

from ingestion.schema import WEATHER_SCHEMA

def load_config():
    """
    Reads settings.yaml and returns it as a Python dictionary.
    All configuration values come from here — nothing is hardcoded.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "settings.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_weather_data(config):
    """
    Calls the Open-Meteo archive API with parameters from config.
    Returns a Pandas DataFrame with one row per day.
    """
    api = config["api"]

    params = {
        "latitude":   api["latitude"],
        "longitude":  api["longitude"],
        "start_date": api["start_date"],
        "end_date":   api["end_date"],
        "daily":      ",".join(api["variables"]),
        "timezone":   api["timezone"],
    }

    print(f"Calling Open-Meteo API for {api['city']}...")
    response = requests.get(api["base_url"], params=params)

    # Fail loudly if the API returns an error
    response.raise_for_status()

    data = response.json()
    daily = data["daily"]

    # Build a DataFrame from the parallel arrays in the response
    # Each index position represents one day
    df = pd.DataFrame({
        "date":            daily["time"],
        "city":            api["city"],
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitation_sum": daily["precipitation_sum"],
        "wind_speed_max":  daily["wind_speed_10m_max"],
        "weather_code":    daily["weather_code"],
    })

    print(f"Fetched {len(df)} rows for {api['city']}")
    return df


def validate_and_convert(df):
    """
    Enforces the explicit PyArrow schema against the DataFrame.
    Converts it to a PyArrow Table — the format needed for Parquet.
    Fails loudly if any column does not match the expected schema.
    """
    print("Validating data against schema...")

    # Check for nulls in non-nullable columns before conversion
    non_nullable = ["date", "city", "temperature_max", "temperature_min"]
    for col in non_nullable:
        if df[col].isnull().any():
            raise ValueError(
                f"Column '{col}' contains null values. "
                f"This is a critical data quality failure."
            )

    # Convert DataFrame to PyArrow Table with explicit schema enforcement
    table = pa.Table.from_pandas(df, schema=WEATHER_SCHEMA, preserve_index=False)
    print(f"Schema validation passed. {table.num_rows} rows validated.")
    return table


def upload_to_gcs(table, config):
    """
    Writes the PyArrow Table as a Parquet file locally,
    then uploads it to the GCS bronze bucket.
    """
    gcp    = config["gcp"]
    api    = config["api"]
    pipeline = config["pipeline"]

    # Set GCP credentials from the JSON key file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    # Define the local temp path and the GCS destination path
    local_path = "temp_weather.parquet"
    gcs_path   = (
        f"{pipeline['bronze_prefix']}/"
        f"weather_{api['start_date']}_{api['end_date']}.parquet"
    )

    # Write Parquet file locally first
    print("Writing Parquet file locally...")
    pq.write_table(table, local_path, compression="snappy")

    # Upload to GCS
    print(f"Uploading to GCS: gs://{gcp['bucket_name']}/{gcs_path}")
    client = storage.Client(project=gcp["project_id"])
    bucket = client.bucket(gcp["bucket_name"])
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

    # Clean up local temp file
    os.remove(local_path)
    print(f"Upload complete. Raw file is now in bronze layer.")


def main():
    config = load_config()
    df     = fetch_weather_data(config)
    table  = validate_and_convert(df)
    upload_to_gcs(table, config)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()