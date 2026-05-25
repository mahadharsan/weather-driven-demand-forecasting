# forecasting/train.py
# Trains five demand forecasting models on mart_demand_features
# Compares naive baseline, Prophet, and XGBoost with and without weather
# Weather features use behavioral demand signals not raw meteorological values
# Tomorrow's weather uses next-day observed data as proxy for forecast API
# In production, tomorrow's features would come from Open-Meteo forecast API

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import bigquery
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pull_data_from_bigquery(config):
    """
    Pulls mart_demand_features from BigQuery into a Pandas DataFrame.
    This is the single source of truth for the forecasting model.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    client = bigquery.Client(project=gcp["project_id"])

    query = """
        SELECT *
        FROM `climate-de-pipeline.weather_marts.mart_demand_features`
        ORDER BY date ASC
    """

    print("Pulling data from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"Pulled {len(df)} rows from mart_demand_features")
    return df


def prepare_data(df):
    """
    Cleans and prepares the DataFrame for modeling.
    Drops rows with null lag features (first 14 rows).
    Adds Option E forward-looking weather proxy features.
    Monthly anomaly leakage fix applied in add_weather_shock_features().
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['sales_lag_7', 'sales_lag_14']).copy()
    df = df.reset_index(drop=True)

    # Option E — Tomorrow's weather as proxy for forecast API
    # In production these would come from Open-Meteo forecast API
    # For backtesting, next-day observed weather is the perfect proxy
    df['tomorrow_temp']           = df['temperature_max'].shift(-1)
    df['tomorrow_precipitation']  = df['precipitation_sum'].shift(-1)
    df['tomorrow_extreme_heat']   = df['is_extreme_heat'].shift(-1)
    df['tomorrow_heavy_rain']     = df['is_heavy_rain'].shift(-1)

    # Sudden temperature change — humans react to unexpected shifts
    df['temp_change_3d'] = df['temperature_max'] - df['temperature_max'].shift(3)

    print(f"After cleaning: {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def add_weather_shock_features(df, train):
    """
    Adds Option D weather shock features.
    Monthly averages computed from training set only to avoid leakage.
    """
    # Monthly temperature baseline from training data only
    monthly_means = (
        train.groupby(train['date'].dt.month)['temperature_max'].mean()
    )
    df['month'] = df['date'].dt.month
    df['temp_anomaly_from_monthly'] = (
        df['temperature_max'] - df['month'].map(monthly_means)
    )

    # Unseasonable heat flag — hot day relative to monthly normal
    # Sparse binary flag helps XGBoost isolate rare demand spikes
    df['unseasonable_heat'] = (
        df['temp_anomaly_from_monthly'] > 8
    ).astype(int)

    # Heuristic retail shopping deterrence index
    # Higher score = worse weather for store visits
    # WMO codes: 2,3=cloudy/overcast; 45,48=fog; 51-65=drizzle/rain
    df['shopping_deterrence'] = (
        (df['precipitation_sum'] > 2).astype(int) * 2 +
        (df['temperature_max'] < 15).astype(int) * 1 +
        (df['weather_code'].isin(
            [2, 3, 45, 48, 51, 53, 55, 61, 63, 65]
        )).astype(int) * 1
    )

    return df


def split_data(df):
    """
    Splits data chronologically into train and test sets.
    Never shuffle time series data.
    Train: 2011 to 2015
    Test:  2016 onwards
    """
    train = df[df['date'] < '2016-01-01'].copy()
    test  = df[df['date'] >= '2016-01-01'].copy()

    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    return train, test


def model_naive(train, test):
    """
    Naive baseline: predict tomorrow = last known value.
    This is the floor — every other model must beat this.
    """
    print("\nTraining Naive baseline...")
    last_value = train['units_sold'].iloc[-1]
    predictions = [last_value] * len(test)
    return predictions


def model_prophet_no_weather(train, test):
    """
    Prophet without weather features.
    Uses only historical sales patterns — trend and seasonality.
    """
    print("\nTraining Prophet without weather...")

    prophet_train = train[['date', 'units_sold']].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    model.fit(prophet_train)

    future = pd.DataFrame({'ds': test['date']})
    forecast = model.predict(future)
    predictions = forecast['yhat'].values

    return predictions


def model_prophet_with_weather(train, test):
    """
    Prophet with weather regressors.
    Uses behavioral weather signals as external regressors.
    """
    print("\nTraining Prophet with weather...")

    weather_regressors = [
        'temperature_max', 'precipitation_sum',
        'temp_anomaly_from_monthly', 'shopping_deterrence',
        'unseasonable_heat', 'tomorrow_temp',
        'tomorrow_precipitation'
    ]

    for col in weather_regressors:
        median_val = train[col].median()
        train[col] = train[col].fillna(median_val).astype(float)
        test[col]  = test[col].fillna(median_val).astype(float)

    prophet_train = train[['date', 'units_sold'] + weather_regressors].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )

    for regressor in weather_regressors:
        model.add_regressor(regressor)

    model.fit(prophet_train)

    future = test[['date'] + weather_regressors].rename(columns={'date': 'ds'})
    forecast = model.predict(future)
    predictions = forecast['yhat'].values

    return predictions


def model_xgboost_no_weather(train, test):
    """
    XGBoost without weather features.
    Uses only sales history and calendar features.
    Baseline for measuring weather signal value.
    """
    print("\nTraining XGBoost without weather...")

    base_features = [
        'sales_lag_7', 'sales_lag_14',
        'sales_rolling_avg_7d', 'sales_rolling_avg_14d',
        'day_of_week', 'snap_ca', 'is_event'
    ]

    X_train = train[base_features].fillna(0)
    y_train = train['units_sold']
    X_test  = test[base_features].fillna(0)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return predictions


def model_xgboost_with_weather(train, test):
    """
    XGBoost with behavioral weather shock features + forward-looking weather.
    Option D: weather shocks (anomaly, deterrence, unseasonable heat)
    Option E: tomorrow's weather as proxy for forecast API
    """
    print("\nTraining XGBoost with weather...")

    weather_features = [
        # Sales history
        'sales_lag_7', 'sales_lag_14',
        'sales_rolling_avg_7d', 'sales_rolling_avg_14d',
        'day_of_week', 'snap_ca', 'is_event',

        # Raw weather
        'temperature_max', 'precipitation_sum',
        'is_extreme_heat', 'is_heavy_rain',

        # Option D — weather shock features
        'temp_anomaly_from_monthly',  # deviation from monthly normal
        'temp_change_3d',              # sudden temperature shift
        'shopping_deterrence',         # heuristic store visit deterrence
        'unseasonable_heat',           # rare hot day flag

        # Option E — forward-looking weather proxy
        'tomorrow_temp',
        'tomorrow_precipitation',
        'tomorrow_extreme_heat',
        'tomorrow_heavy_rain'
    ]

    X_train = train[weather_features].fillna(0)
    y_train = train['units_sold']
    X_test  = test[weather_features].fillna(0)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Feature importance — shows which features the model actually used
    importance = pd.DataFrame({
        'feature': weather_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(importance.head(10).to_string(index=False))

    return predictions, model, weather_features, X_test


def calculate_metrics(actuals, predictions, model_name, anomaly_mask=None):
    """
    Calculates RMSE, MAE, MAPE for all days.
    Also calculates MAPE on anomaly days specifically if mask provided.
    Anomaly days are where weather impact is strongest.
    """
    actuals     = np.array(actuals)
    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae  = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\n{model_name}:")
    print(f"  Overall RMSE: {rmse:.1f}")
    print(f"  Overall MAE:  {mae:.1f}")
    print(f"  Overall MAPE: {mape:.2f}%")

    if anomaly_mask is not None and anomaly_mask.sum() > 0:
        anomaly_actuals = actuals[anomaly_mask]
        anomaly_preds   = predictions[anomaly_mask]
        anomaly_mape    = np.mean(
            np.abs((anomaly_actuals - anomaly_preds) / anomaly_actuals)
        ) * 100
        print(f"  Anomaly Day MAPE: {anomaly_mape:.2f}% ({anomaly_mask.sum()} days)")

    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}


def save_predictions_to_bigquery(test, all_predictions, config):
    """
    Saves all model predictions alongside actuals to BigQuery.
    """
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]

    output_df = pd.DataFrame({
        'date':                    test['date'].dt.strftime('%Y-%m-%d'),
        'actual_units_sold':       test['units_sold'].values,
        'naive_prediction':        all_predictions['naive'],
        'prophet_no_weather':      all_predictions['prophet_no_weather'],
        'prophet_with_weather':    all_predictions['prophet_with_weather'],
        'xgb_no_weather':          all_predictions['xgb_no_weather'],
        'xgb_with_weather':        all_predictions['xgb_with_weather'],
    })

    table_id = f"{gcp['project_id']}.weather_marts.mart_forecast_output"
    client   = bigquery.Client(project=gcp["project_id"])

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )

    client.load_table_from_dataframe(
        output_df, table_id, job_config=job_config
    ).result()

    print(f"\nPredictions saved to {table_id}")
    print(f"Rows saved: {len(output_df)}")


def main():
    config = load_config()

    # Pull and prepare data
    df          = pull_data_from_bigquery(config)
    df          = prepare_data(df)
    train, test = split_data(df)

    # Add weather shock features using training-set-only monthly averages
    # This prevents temporal leakage from future data
    train = add_weather_shock_features(train, train)
    test  = add_weather_shock_features(test, train)

    # Anomaly mask for targeted evaluation
    anomaly_mask = test['anomaly_flag'].values == 1
    print(f"\nAnomaly days in test set: {anomaly_mask.sum()}")

    # Train all five models
    naive_preds                              = model_naive(train, test)
    prophet_no_weather                       = model_prophet_no_weather(train, test)
    prophet_with_weather                     = model_prophet_with_weather(train, test)
    xgb_no_weather                           = model_xgboost_no_weather(train, test)
    xgb_with_weather, xgb_model, feats, X_t  = model_xgboost_with_weather(train, test)

    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)

    actuals = test['units_sold'].values

    metrics = [
        calculate_metrics(actuals, naive_preds,
                         "Naive Baseline", anomaly_mask),
        calculate_metrics(actuals, prophet_no_weather,
                         "Prophet (no weather)", anomaly_mask),
        calculate_metrics(actuals, prophet_with_weather,
                         "Prophet (with weather)", anomaly_mask),
        calculate_metrics(actuals, xgb_no_weather,
                         "XGBoost (no weather)", anomaly_mask),
        calculate_metrics(actuals, xgb_with_weather,
                         "XGBoost (with weather)", anomaly_mask),
    ]

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print("-"*58)
    for m in metrics:
        print(
            f"{m['model']:<30} {m['rmse']:>8.1f} "
            f"{m['mae']:>8.1f} {m['mape']:>7.2f}%"
        )

    # Save predictions
    all_predictions = {
        'naive':               naive_preds,
        'prophet_no_weather':  prophet_no_weather,
        'prophet_with_weather': prophet_with_weather,
        'xgb_no_weather':      xgb_no_weather,
        'xgb_with_weather':    xgb_with_weather,
    }

    save_predictions_to_bigquery(test, all_predictions, config)
    print("\nForecasting pipeline complete.")


if __name__ == "__main__":
    main()