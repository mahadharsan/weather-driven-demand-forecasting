# orchestration/resources.py
# Defines reusable GCP connections for the pipeline
# Resources are injected into assets that need them
# This means credentials and clients are configured once, not repeated

import os
from dagster import resource
from google.cloud import storage, bigquery


def get_gcp_clients(credentials_path: str, project_id: str):
    """
    Returns GCS and BigQuery clients authenticated
    using the service account credentials file.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    gcs_client = storage.Client(project=project_id)
    bq_client  = bigquery.Client(project=project_id)
    
    return gcs_client, bq_client