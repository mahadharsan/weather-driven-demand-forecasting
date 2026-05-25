# forecasting/visualize.py
# Generates visualization of actual vs predicted sales
# Highlights anomaly days to show weather model improvement
# Output saved to docs/forecast_comparison.png

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from google.cloud import bigquery


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pull_forecast_data(config):
    """
    Pulls forecast output and weather data from BigQuery.
    Joins to get anomaly flags for highlighting.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]
    client = bigquery.Client(project=gcp["project_id"])

    query = """
        SELECT
            f.date,
            f.actual_units_sold,
            f.xgb_no_weather,
            f.xgb_with_weather,
            w.anomaly_flag,
            w.weather_description,
            w.temperature_max,
            w.precipitation_sum
        FROM `climate-de-pipeline.weather_marts.mart_forecast_output` f
        LEFT JOIN `climate-de-pipeline.weather_marts.mart_weather_daily` w
            ON CAST(f.date AS DATE) = w.date
        ORDER BY f.date ASC
    """

    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df


def plot_forecast_comparison(df):
    """
    Creates a clean forecast comparison chart showing:
    - Actual sales
    - XGBoost without weather
    - XGBoost with weather
    - Anomaly days highlighted
    """

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        'Weather-Driven Demand Forecasting — CA_3 Walmart FOODS\n'
        'XGBoost with Behavioral Weather Signals vs Baseline',
        fontsize=14, fontweight='bold', y=0.98
    )

    # --- Top chart: Full forecast period ---
    ax1 = axes[0]

    ax1.plot(df['date'], df['actual_units_sold'],
             color='#2c3e50', linewidth=1.5, label='Actual Sales', zorder=3)
    ax1.plot(df['date'], df['xgb_no_weather'],
             color='#e74c3c', linewidth=1.2, linestyle='--',
             label='XGBoost (no weather)', alpha=0.8, zorder=2)
    ax1.plot(df['date'], df['xgb_with_weather'],
             color='#27ae60', linewidth=1.2, linestyle='-.',
             label='XGBoost (with weather)', alpha=0.8, zorder=2)

    # Highlight anomaly days
    anomaly_days = df[df['anomaly_flag'] == 1]
    for _, row in anomaly_days.iterrows():
        ax1.axvline(x=row['date'], color='#f39c12', linewidth=2,
                   alpha=0.7, zorder=1)
        ax1.annotate(
            f"{row['temperature_max']:.0f}°C\n{row['weather_description']}",
            xy=(row['date'], row['actual_units_sold']),
            xytext=(10, 20), textcoords='offset points',
            fontsize=7, color='#f39c12',
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1)
        )

    anomaly_patch = mpatches.Patch(
        color='#f39c12', alpha=0.7,
        label=f'Anomaly Days (n={len(anomaly_days)})'
    )

    ax1.set_title('Full Test Period: Jan 2016 — May 2016', fontsize=11)
    ax1.set_ylabel('Daily Units Sold')
    ax1.legend(handles=[
        ax1.lines[0], ax1.lines[1], ax1.lines[2], anomaly_patch
    ], loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # --- Bottom chart: Error comparison ---
    ax2 = axes[1]

    error_no_weather   = np.abs(df['actual_units_sold'] - df['xgb_no_weather'])
    error_with_weather = np.abs(df['actual_units_sold'] - df['xgb_with_weather'])

    ax2.fill_between(df['date'], error_no_weather,
                     alpha=0.4, color='#e74c3c',
                     label='Error — no weather')
    ax2.fill_between(df['date'], error_with_weather,
                     alpha=0.4, color='#27ae60',
                     label='Error — with weather')

    # Highlight anomaly days on error chart
    for _, row in anomaly_days.iterrows():
        ax2.axvline(x=row['date'], color='#f39c12',
                   linewidth=2, alpha=0.7)

    ax2.set_title(
        'Absolute Forecast Error — Weather Model Reduces Error on Anomaly Days',
        fontsize=11
    )
    ax2.set_ylabel('Absolute Error (Units)')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    plt.tight_layout()

    # Save to docs folder
    output_path = Path(__file__).parent.parent / "docs" / "forecast_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Chart saved to {output_path}")
    plt.show()


def main():
    config = load_config()
    df     = pull_forecast_data(config)
    print(f"Pulled {len(df)} forecast rows")
    print(f"Anomaly days: {df['anomaly_flag'].sum()}")
    plot_forecast_comparison(df)


if __name__ == "__main__":
    main()