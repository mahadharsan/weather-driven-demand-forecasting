# loading/load_to_bq.py
# Sole responsibility: load the raw Parquet file from GCS
# into BigQuery as a raw table in the weather_raw dataset
# Does NOT transform. Loads exactly what is in the bronze layer.

import os
import yaml
from google.cloud import bigquery


def load_config():
    """
    Reads settings.yaml and returns it as a Python dictionary.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "settings.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_parquet_to_bigquery(config):
    """
    Loads the raw Parquet file from GCS into BigQuery.
    Uses WRITE_TRUNCATE — meaning every run replaces the table completely.
    This ensures idempotency: running twice gives the same result as once.
    """
    gcp      = config["gcp"]
    api      = config["api"]
    pipeline = config["pipeline"]

    # Set credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    # Build the GCS URI — where the Parquet file lives
    gcs_uri = (
        f"gs://{gcp['bucket_name']}/"
        f"{pipeline['bronze_prefix']}/"
        f"weather_{api['start_date']}_{api['end_date']}.parquet"
    )

    # Build the BigQuery table reference
    table_id = (
        f"{gcp['project_id']}."
        f"{gcp['dataset_id']}."
        f"{pipeline['raw_table']}"
    )

    print(f"Loading from: {gcs_uri}")
    print(f"Loading into: {table_id}")

    # Initialize BigQuery client
    client = bigquery.Client(project=gcp["project_id"])

    # Define the load job configuration
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=False,
        schema=[
            bigquery.SchemaField("date",             "STRING",  mode="REQUIRED"),
            bigquery.SchemaField("city",             "STRING",  mode="REQUIRED"),
            bigquery.SchemaField("temperature_max",  "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("temperature_min",  "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("precipitation_sum","FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("wind_speed_max",   "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("weather_code",     "INTEGER", mode="NULLABLE"),
        ]
    )

    # Run the load job
    load_job = client.load_table_from_uri(
        gcs_uri,
        table_id,
        job_config=job_config
    )

    # Wait for the job to complete
    load_job.result()

    # Verify
    table = client.get_table(table_id)
    print(f"Load complete. {table.num_rows} rows now in {table_id}")


def main():
    config = load_config()
    load_parquet_to_bigquery(config)
    print("BigQuery load complete.")


if __name__ == "__main__":
    main()