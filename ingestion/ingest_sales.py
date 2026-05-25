# ingestion/ingest_sales.py
# Sole responsibility: transform M5 Walmart sales data for CA_3 FOODS
# into daily aggregated time series and upload to GCS bronze layer

import os
import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from google.cloud import storage


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_transform_sales():
    """
    Loads M5 sales and calendar CSVs.
    Filters for CA_3 FOODS only.
    Melts from wide to long format.
    Joins with calendar to get real dates.
    Aggregates to daily total sales.
    """
    base_path = Path(__file__).parent.parent / "data" / "raw" / "m5"

    print("Loading sales data...")
    sales = pd.read_csv(base_path / "sales_train_evaluation.csv")

    print("Loading calendar data...")
    calendar = pd.read_csv(base_path / "calendar.csv")

    # Filter for CA_3 FOODS only
    print("Filtering for CA_3 FOODS...")
    ca3_foods = sales[
        (sales['store_id'] == 'CA_3') &
        (sales['cat_id'] == 'FOODS')
    ].copy()

    print(f"CA_3 FOODS items: {len(ca3_foods)}")

    # Get all day columns
    day_cols = [col for col in ca3_foods.columns if col.startswith('d_')]

    # Melt from wide to long format
    print("Melting to long format...")
    melted = ca3_foods.melt(
        id_vars=['item_id', 'store_id'],
        value_vars=day_cols,
        var_name='d',
        value_name='units_sold'
    )

    # Create d column in calendar
    calendar['d'] = ['d_' + str(i + 1) for i in range(len(calendar))]

    # Join with calendar to get real dates and features
    print("Joining with calendar...")
    merged = melted.merge(
        calendar[['d', 'date', 'snap_CA', 'event_name_1', 'wday']],
        on='d',
        how='left'
    )

    # Create binary event flag
    merged['is_event'] = merged['event_name_1'].notna().astype(int)

    # Aggregate to daily total
    print("Aggregating to daily totals...")
    daily = merged.groupby('date').agg(
        units_sold=('units_sold', 'sum'),
        snap_ca=('snap_CA', 'first'),
        is_event=('is_event', 'first'),
        day_of_week=('wday', 'first')
    ).reset_index()

    # Sort by date
    daily = daily.sort_values('date').reset_index(drop=True)

    print(f"Daily sales rows: {len(daily)}")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"Sample:\n{daily.head()}")

    return daily


def validate_and_convert(df):
    """
    Validates the DataFrame and converts to PyArrow Table.
    """
    print("Validating schema...")

    # Check for nulls in critical columns
    for col in ['date', 'units_sold']:
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains null values.")

    schema = pa.schema([
        pa.field('date',        pa.string(),  nullable=False),
        pa.field('units_sold',  pa.int64(),   nullable=False),
        pa.field('snap_ca',     pa.int64(),   nullable=True),
        pa.field('is_event',    pa.int64(),   nullable=True),
        pa.field('day_of_week', pa.int64(),   nullable=True),
    ])

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    print(f"Schema validation passed. {table.num_rows} rows validated.")
    return table


def upload_to_gcs(table, config):
    """
    Uploads the sales Parquet file to GCS bronze layer.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    local_path  = "temp_sales.parquet"
    gcs_path    = "bronze/sales/ca3_foods_daily.parquet"

    print("Writing Parquet file locally...")
    pq.write_table(table, local_path, compression="snappy")

    print(f"Uploading to GCS: gs://{gcp['bucket_name']}/{gcs_path}")
    client = storage.Client(project=gcp["project_id"])
    bucket = client.bucket(gcp["bucket_name"])
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

    os.remove(local_path)
    print("Upload complete. Sales data now in bronze layer.")


def main():
    config = load_config()
    df     = load_and_transform_sales()
    table  = validate_and_convert(df)
    upload_to_gcs(table, config)
    print("Sales ingestion complete.")


if __name__ == "__main__":
    main()