# orchestration/definitions.py
# Main entry point for Dagster
# Registers all assets, dbt project, and schedule
# Run with: dagster dev -f orchestration/definitions.py

from dagster import Definitions, ScheduleDefinition, define_asset_job
from dagster_dbt import DbtCliResource, dbt_assets, DbtProject
from pathlib import Path
from orchestration.assets import raw_weather_parquet, raw_weather_bq_table

# Point Dagster to dbt project folder
dbt_project = DbtProject(
    project_dir=Path(__file__).parent.parent / "dbt"
)

# This single decorator reads dbt manifest and creates
# Dagster assets from ALL dbt models automatically
# stg_weather and mart_weather_daily become assets with zero extra code
@dbt_assets(manifest=dbt_project.manifest_path)
def weather_dbt_assets(context, dbt: DbtCliResource):
    yield from dbt.cli(["run"], context=context).stream()

# A job that runs all four assets in dependency order
weather_pipeline_job = define_asset_job(
    name="weather_pipeline_job",
    selection="*"
)

# Runs automatically every day at 6am
daily_schedule = ScheduleDefinition(
    job=weather_pipeline_job,
    cron_schedule="0 6 * * *",
    name="daily_weather_schedule"
)

# Register everything with Dagster
defs = Definitions(
    assets=[
        raw_weather_parquet,
        raw_weather_bq_table,
        weather_dbt_assets
    ],
    resources={
        "dbt": DbtCliResource(
            project_dir=str(dbt_project.project_dir)
        )
    },
    schedules=[daily_schedule]
)