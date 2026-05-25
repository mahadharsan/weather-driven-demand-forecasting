# loading/load_sales_to_bq.py
# Sole responsibility: load the raw sales Parquet file from GCS
# into BigQuery as a raw table in the sales_raw dataset

import os
import yaml
from pathlib import Path
from google.cloud import bigquery


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sales_to_bigquery(config):
    """
    Loads the CA_3 FOODS daily sales Parquet file from GCS
    into BigQuery sales_raw dataset.
    """
    gcp = config["gcp"]

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    gcs_uri  = f"gs://{gcp['bucket_name']}/bronze/sales/ca3_foods_daily.parquet"
    table_id = f"{gcp['project_id']}.sales_raw.sales_raw_ca3_foods"

    print(f"Loading from: {gcs_uri}")
    print(f"Loading into: {table_id}")

    client = bigquery.Client(project=gcp["project_id"])

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=False,
        schema=[
            bigquery.SchemaField("date",        "STRING",  mode="REQUIRED"),
            bigquery.SchemaField("units_sold",  "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("snap_ca",     "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("is_event",    "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("day_of_week", "INTEGER", mode="NULLABLE"),
        ]
    )

    load_job = client.load_table_from_uri(
        gcs_uri,
        table_id,
        job_config=job_config
    )

    load_job.result()

    table = client.get_table(table_id)
    print(f"Load complete. {table.num_rows} rows now in {table_id}")


def main():
    config = load_config()
    load_sales_to_bigquery(config)
    print("Sales BigQuery load complete.")


if __name__ == "__main__":
    main()