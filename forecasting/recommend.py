# forecasting/recommend.py
# Interactive AI-powered inventory recommendation system
# User inputs dates, system generates operational recommendations
# Uses Google Gemini API (free tier) — gemini-2.0-flash

import os
import time
import yaml
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from google import genai

load_dotenv()


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pull_data_for_dates(config, dates):
    """
    Pulls forecast and weather data for specific dates from BigQuery.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]
    client = bigquery.Client(project=gcp["project_id"])

    date_list = ", ".join([f"'{d}'" for d in dates])

    query = f"""
        SELECT
            f.date,
            f.actual_units_sold,
            f.xgb_with_weather AS forecasted_units,
            ABS(f.actual_units_sold - f.xgb_with_weather)
                / f.actual_units_sold * 100 AS error_pct,
            w.temperature_max,
            w.precipitation_sum,
            w.anomaly_flag,
            w.weather_description,
            w.season,
            w.is_extreme_heat,
            w.is_heavy_rain,
            d.snap_ca,
            d.is_event,
            d.day_of_week,
            d.sales_rolling_avg_7d
        FROM `climate-de-pipeline.weather_marts.mart_forecast_output` f
        LEFT JOIN `climate-de-pipeline.weather_marts.mart_weather_daily` w
            ON CAST(f.date AS DATE) = w.date
        LEFT JOIN `climate-de-pipeline.weather_marts.mart_demand_features` d
            ON CAST(f.date AS DATE) = d.date
        WHERE CAST(f.date AS STRING) IN ({date_list})
        ORDER BY f.date ASC
    """

    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_available_dates(config):
    """
    Shows user what dates are available in the forecast output.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]
    client = bigquery.Client(project=gcp["project_id"])

    query = """
        SELECT 
            MIN(date) as start_date,
            MAX(date) as end_date,
            COUNT(*) as total_days
        FROM `climate-de-pipeline.weather_marts.mart_forecast_output`
    """

    result = client.query(query).to_dataframe()
    return result.iloc[0]


def build_prompt(row, rolling_avg):
    day_names = {
        1: 'Saturday', 2: 'Sunday', 3: 'Monday',
        4: 'Tuesday', 5: 'Wednesday', 6: 'Thursday', 7: 'Friday'
    }

    demand_vs_avg = (
        (float(row['forecasted_units']) - rolling_avg) / rolling_avg * 100
        if rolling_avg > 0 else 0
    )

    prompt = f"""You are a supply chain intelligence system for a Walmart FOODS 
department in Los Angeles, California.

Generate an operational inventory recommendation based on this data:

DATE: {row['date'].date()} ({day_names.get(int(row['day_of_week']), 'Unknown')})

FORECAST:
- Forecasted demand: {int(row['forecasted_units'])} units
- Actual demand: {int(row['actual_units_sold'])} units
- Demand vs 7-day average: {demand_vs_avg:+.1f}%

WEATHER:
- Temperature: {float(row['temperature_max'])}°C
- Precipitation: {float(row['precipitation_sum'])}mm
- Conditions: {row['weather_description']}
- Season: {row['season']}
- Weather anomaly: {bool(row['anomaly_flag'])}
- Extreme heat: {bool(row['is_extreme_heat'])}
- Heavy rain: {bool(row['is_heavy_rain'])}

CALENDAR:
- SNAP benefits active: {bool(row['snap_ca'])}
- Special event: {bool(row['is_event'])}

Generate recommendation in exactly this format:

DEMAND OUTLOOK: [1 sentence]

PRIMARY DRIVERS:
- [driver 1]
- [driver 2]
- [driver 3]

WEATHER IMPACT: [1 sentence on weather effect]

INVENTORY ACTION: [specific actionable recommendation]

RISK LEVEL: [LOW / MODERATE / ELEVATED / HIGH] — [1 sentence reason]

Keep under 150 words. Be specific and operational."""

    return prompt


def generate_recommendation(row, gemini_client):
    """
    Generates recommendation for a single day with retry logic.
    """
    rolling_avg = (
        float(row['sales_rolling_avg_7d'])
        if pd.notna(row['sales_rolling_avg_7d'])
        else float(row['forecasted_units'])
    )

    prompt = build_prompt(row, rolling_avg)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds... ({attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                return f"[Recommendation unavailable: {str(e)}]"


def get_user_dates(available):
    """
    Interactive prompt asking user which dates to generate recommendations for.
    """
    print(f"\nAvailable forecast dates: {available['start_date']} to {available['end_date']}")
    print(f"Total days available: {int(available['total_days'])}")
    print("\nEnter dates you want recommendations for.")
    print("Format: YYYY-MM-DD (comma separated for multiple)")
    print("Example: 2016-02-15, 2016-03-04, 2016-04-06")
    print("Or type 'anomaly' to get recommendations for anomaly days only")
    print("Or type 'snap' to get recommendations for SNAP benefit days")

    user_input = input("\nEnter dates: ").strip()
    return user_input


def resolve_dates(user_input, config):
    """
    Resolves user input to a list of date strings.
    Handles specific dates, 'anomaly', and 'snap' keywords.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]
    client = bigquery.Client(project=gcp["project_id"])

    if user_input.lower() == 'anomaly':
        query = """
            SELECT CAST(f.date AS STRING) as date
            FROM `climate-de-pipeline.weather_marts.mart_forecast_output` f
            LEFT JOIN `climate-de-pipeline.weather_marts.mart_weather_daily` w
                ON CAST(f.date AS DATE) = w.date
            WHERE w.anomaly_flag = 1
            ORDER BY f.date
        """
        df = client.query(query).to_dataframe()
        return df['date'].tolist()

    elif user_input.lower() == 'snap':
        query = """
            SELECT CAST(f.date AS STRING) as date
            FROM `climate-de-pipeline.weather_marts.mart_forecast_output` f
            LEFT JOIN `climate-de-pipeline.weather_marts.mart_demand_features` d
                ON CAST(f.date AS DATE) = d.date
            WHERE d.snap_ca = 1
            ORDER BY f.date
            LIMIT 5
        """
        df = client.query(query).to_dataframe()
        return df['date'].tolist()

    else:
        dates = [d.strip() for d in user_input.split(',')]
        return dates


def main():
    print("="*70)
    print("Weather-Driven Demand Forecasting — AI Recommendation System")
    print("Powered by Google Gemini (free tier)")
    print("="*70)

    config = load_config()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Add it to your .env file. "
            "Get a free key at aistudio.google.com"
        )

    gemini_client = genai.Client(api_key=api_key)

    print("\nConnecting to BigQuery...")
    available = get_available_dates(config)

    user_input = get_user_dates(available)
    dates      = resolve_dates(user_input, config)

    print(f"\nFetching data for {len(dates)} date(s)...")
    df = pull_data_for_dates(config, dates)

    if df.empty:
        print("No data found for the specified dates.")
        print(f"Available range: {available['start_date']} to {available['end_date']}")
        return

    print(f"Found {len(df)} day(s). Generating recommendations...")

    for _, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"DATE: {row['date'].date()}")
        print(f"Forecast: {int(row['forecasted_units'])} units | "
              f"Actual: {int(row['actual_units_sold'])} units | "
              f"Error: {float(row['error_pct']):.1f}%")
        print(f"Weather: {float(row['temperature_max'])}°C, "
              f"{row['weather_description']}")
        print(f"Anomaly: {bool(row['anomaly_flag'])} | "
              f"SNAP: {bool(row['snap_ca'])} | "
              f"Event: {bool(row['is_event'])}")
        print("-"*70)

        recommendation = generate_recommendation(row, gemini_client)
        print(recommendation)

    print(f"\n{'='*70}")
    print("Recommendation session complete.")


if __name__ == "__main__":
    main()