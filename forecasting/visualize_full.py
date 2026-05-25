# forecasting/visualize_full.py
# Generates complete ML visualization suite for the forecasting project
# Outputs 6 charts to docs/ folder

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from google.cloud import bigquery
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def pull_data(config):
    gcp = config["gcp"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp["credentials_path"]
    client = bigquery.Client(project=gcp["project_id"])

    query = """
        SELECT *
        FROM `climate-de-pipeline.weather_marts.mart_demand_features`
        ORDER BY date ASC
    """
    df = client.query(query).to_dataframe()
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_and_split(df):
    df = df.dropna(subset=['sales_lag_7', 'sales_lag_14']).copy()
    df = df.reset_index(drop=True)

    # Option E forward looking weather proxy
    df['tomorrow_temp']          = df['temperature_max'].shift(-1)
    df['tomorrow_precipitation'] = df['precipitation_sum'].shift(-1)
    df['tomorrow_extreme_heat']  = df['is_extreme_heat'].shift(-1)
    df['tomorrow_heavy_rain']    = df['is_heavy_rain'].shift(-1)
    df['temp_change_3d']         = df['temperature_max'] - df['temperature_max'].shift(3)

    train = df[df['date'] < '2016-01-01'].copy()
    test  = df[df['date'] >= '2016-01-01'].copy()

    # Monthly averages from training set only
    monthly_means = train.groupby(train['date'].dt.month)['temperature_max'].mean()

    for split in [train, test]:
        split['month'] = split['date'].dt.month
        split['temp_anomaly_from_monthly'] = (
            split['temperature_max'] - split['month'].map(monthly_means)
        )
        split['unseasonable_heat'] = (
            split['temp_anomaly_from_monthly'] > 8
        ).astype(int)
        split['shopping_deterrence'] = (
            (split['precipitation_sum'] > 2).astype(int) * 2 +
            (split['temperature_max'] < 15).astype(int) * 1
        )

    return train, test


def retrain_models(train, test):
    """Retrains XGBoost models and Prophet for visualization."""

    base_features = [
        'sales_lag_7', 'sales_lag_14',
        'sales_rolling_avg_7d', 'sales_rolling_avg_14d',
        'day_of_week', 'snap_ca', 'is_event'
    ]

    weather_features = base_features + [
        'temperature_max', 'precipitation_sum',
        'is_extreme_heat', 'is_heavy_rain',
        'temp_anomaly_from_monthly', 'temp_change_3d',
        'shopping_deterrence', 'unseasonable_heat',
        'tomorrow_temp', 'tomorrow_precipitation',
        'tomorrow_extreme_heat', 'tomorrow_heavy_rain'
    ]

    # XGBoost no weather
    xgb_base = XGBRegressor(n_estimators=500, learning_rate=0.05,
                             max_depth=6, subsample=0.8, random_state=42)
    xgb_base.fit(train[base_features].fillna(0), train['units_sold'])
    pred_base = xgb_base.predict(test[base_features].fillna(0))

    # XGBoost with weather
    xgb_weather = XGBRegressor(n_estimators=500, learning_rate=0.05,
                                max_depth=6, subsample=0.8, random_state=42)
    xgb_weather.fit(train[weather_features].fillna(0), train['units_sold'])
    pred_weather = xgb_weather.predict(test[weather_features].fillna(0))

    # Prophet no weather
    prophet_train = train[['date', 'units_sold']].rename(
        columns={'date': 'ds', 'units_sold': 'y'}
    )
    prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                      seasonality_mode='multiplicative')
    prophet.fit(prophet_train)
    future = pd.DataFrame({'ds': test['date']})
    prophet_forecast = prophet.predict(future)
    pred_prophet = prophet_forecast['yhat'].values

    # Naive baseline
    pred_naive = [train['units_sold'].iloc[-1]] * len(test)

    return {
        'naive': pred_naive,
        'prophet': pred_prophet,
        'prophet_forecast': prophet_forecast,
        'xgb_base': pred_base,
        'xgb_weather': pred_weather,
        'xgb_weather_model': xgb_weather,
        'weather_features': weather_features,
        'base_features': base_features
    }


def calc_metrics(actuals, preds):
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    return rmse, mae, mape


# ── Chart 1: Model Comparison Bar Chart ──────────────────────────────────────
def plot_model_comparison(test, preds):
    actuals = test['units_sold'].values
    models  = ['Naive', 'Prophet', 'XGBoost\n(no weather)', 'XGBoost\n(with weather)']
    pred_list = [preds['naive'], preds['prophet'],
                 preds['xgb_base'], preds['xgb_weather']]

    rmses, maes, mapes = [], [], []
    for p in pred_list:
        r, m, mp = calc_metrics(actuals, np.array(p))
        rmses.append(r); maes.append(m); mapes.append(mp)

    x    = np.arange(len(models))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#27ae60']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Model Comparison — All Metrics', fontsize=13, fontweight='bold')

    for ax, values, title, color_base in zip(
        axes,
        [rmses, maes, mapes],
        ['RMSE (lower = better)', 'MAE (lower = better)', 'MAPE % (lower = better)'],
        ['#c0392b', '#2980b9', '#16a085']
    ):
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('docs/1_model_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/1_model_comparison.png")
    plt.close()


# ── Chart 2: Actual vs Predicted Scatter ─────────────────────────────────────
def plot_actual_vs_predicted(test, preds):
    actuals = test['units_sold'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Actual vs Predicted — XGBoost Models', fontsize=13, fontweight='bold')

    for ax, pred, title, color in zip(
        axes,
        [preds['xgb_base'], preds['xgb_weather']],
        ['XGBoost (no weather)', 'XGBoost (with weather)'],
        ['#e74c3c', '#27ae60']
    ):
        ax.scatter(actuals, pred, alpha=0.6, color=color, edgecolors='white',
                   linewidth=0.5, s=50)
        min_val = min(actuals.min(), np.array(pred).min())
        max_val = max(actuals.max(), np.array(pred).max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=1.5, alpha=0.5, label='Perfect prediction')
        _, _, mape = calc_metrics(actuals, np.array(pred))
        ax.set_title(f'{title}\nMAPE: {mape:.2f}%', fontsize=10)
        ax.set_xlabel('Actual Units Sold')
        ax.set_ylabel('Predicted Units Sold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('docs/2_actual_vs_predicted.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/2_actual_vs_predicted.png")
    plt.close()


# ── Chart 3: Residual Distribution ───────────────────────────────────────────
def plot_residuals(test, preds):
    actuals = test['units_sold'].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Residual Distribution by Model', fontsize=13, fontweight='bold')

    model_data = [
        ('Naive Baseline', preds['naive'], '#95a5a6'),
        ('Prophet', preds['prophet'], '#3498db'),
        ('XGBoost (no weather)', preds['xgb_base'], '#e74c3c'),
        ('XGBoost (with weather)', preds['xgb_weather'], '#27ae60'),
    ]

    for ax, (name, pred, color) in zip(axes.flatten(), model_data):
        residuals = actuals - np.array(pred)
        ax.hist(residuals, bins=25, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axvline(residuals.mean(), color='orange', linewidth=1.5,
                   linestyle='-', alpha=0.9, label=f'Mean: {residuals.mean():.0f}')
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('docs/3_residual_distribution.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/3_residual_distribution.png")
    plt.close()


# ── Chart 4: Feature Importance ──────────────────────────────────────────────
def plot_feature_importance(preds):
    model    = preds['xgb_weather_model']
    features = preds['weather_features']

    importance = pd.DataFrame({
        'feature':    features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    colors = ['#27ae60' if 'tomorrow' in f or 'weather' in f or
              'temp_anomaly' in f or 'shopping' in f or 'unseasonable' in f
              else '#3498db' if 'sales' in f or 'snap' in f or 'event' in f
              else '#95a5a6'
              for f in importance['feature']]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance['feature'], importance['importance'],
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.set_title('XGBoost Feature Importance\n(green = weather features, blue = sales/calendar)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Importance Score')
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    for bar, val in zip(bars, importance['importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)

    import matplotlib.patches as mpatches
    weather_patch = mpatches.Patch(color='#27ae60', alpha=0.85, label='Weather features')
    sales_patch   = mpatches.Patch(color='#3498db', alpha=0.85, label='Sales/calendar features')
    ax.legend(handles=[weather_patch, sales_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig('docs/4_feature_importance.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/4_feature_importance.png")
    plt.close()


# ── Chart 5: Prophet Decomposition ───────────────────────────────────────────
def plot_prophet_decomposition(preds):
    forecast = preds['prophet_forecast']

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Prophet Model Decomposition\nTrend + Weekly + Yearly Seasonality',
                 fontsize=13, fontweight='bold')

    forecast['ds'] = pd.to_datetime(forecast['ds'])

    axes[0].plot(forecast['ds'], forecast['trend'],
                 color='#2c3e50', linewidth=1.5)
    axes[0].set_title('Trend', fontsize=10)
    axes[0].set_ylabel('Units')
    axes[0].grid(alpha=0.3)
    axes[0].set_facecolor('#f8f9fa')

    axes[1].plot(forecast['ds'], forecast['weekly'],
                 color='#3498db', linewidth=1.5)
    axes[1].set_title('Weekly Seasonality', fontsize=10)
    axes[1].set_ylabel('Effect')
    axes[1].grid(alpha=0.3)
    axes[1].set_facecolor('#f8f9fa')

    axes[2].plot(forecast['ds'], forecast['yearly'],
                 color='#e67e22', linewidth=1.5)
    axes[2].set_title('Yearly Seasonality', fontsize=10)
    axes[2].set_ylabel('Effect')
    axes[2].set_xlabel('Date')
    axes[2].grid(alpha=0.3)
    axes[2].set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('docs/5_prophet_decomposition.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/5_prophet_decomposition.png")
    plt.close()


# ── Chart 6: Train/Test Split ─────────────────────────────────────────────────
def plot_train_test_split(train, test):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(train['date'], train['units_sold'],
            color='#3498db', linewidth=1.0, alpha=0.8, label='Training Data (2011-2015)')
    ax.plot(test['date'], test['units_sold'],
            color='#e74c3c', linewidth=1.5, alpha=0.9, label='Test Data (2016)')

    ax.axvline(pd.Timestamp('2016-01-01'), color='black',
               linewidth=2, linestyle='--', alpha=0.7, label='Train/Test Split')
    ax.fill_between(train['date'], train['units_sold'],
                    alpha=0.1, color='#3498db')
    ax.fill_between(test['date'], test['units_sold'],
                    alpha=0.15, color='#e74c3c')

    ax.set_title('Chronological Train/Test Split\nCA_3 Walmart FOODS Daily Sales 2011-2016',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Units Sold')
    ax.set_xlabel('Date')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    train_size = len(train)
    test_size  = len(test)
    ax.text(0.02, 0.95, f'Train: {train_size} days',
            transform=ax.transAxes, fontsize=10, color='#3498db', fontweight='bold')
    ax.text(0.85, 0.95, f'Test: {test_size} days',
            transform=ax.transAxes, fontsize=10, color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    plt.savefig('docs/6_train_test_split.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: docs/6_train_test_split.png")
    plt.close()


def main():
    print("Pulling data from BigQuery...")
    config      = load_config()
    df          = pull_data(config)
    train, test = prepare_and_split(df)

    print("Retraining models for visualization...")
    preds = retrain_models(train, test)

    print("\nGenerating visualizations...")
    plot_train_test_split(train, test)
    plot_model_comparison(test, preds)
    plot_actual_vs_predicted(test, preds)
    plot_residuals(test, preds)
    plot_feature_importance(preds)
    plot_prophet_decomposition(preds)

    print("\nAll 6 charts saved to docs/")


if __name__ == "__main__":
    main()