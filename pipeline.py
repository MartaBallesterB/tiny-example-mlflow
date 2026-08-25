import numpy as np
import polars as pl
#import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

# explicit local DB for mlflow tracking + xgboost autologging
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experiment_1_flight_demand_forecasting")
mlflow.xgboost.autolog(log_models=True)

# mocked data with dates and passenger counts + feature eng
dates = pl.date_range(
    start=pl.date(2025, 1, 1),
    end=pl.date(2025, 12, 31),
    interval="1d",
    eager=True
)
df = pl.DataFrame({
    "date": dates,
    "passenger_count": pl.int_range(100, 300, eager=True).sample(n=len(dates), with_replacement=True)
}).sort("date") # to keep the time series order!

def create_features(df_raw, include_cyclical = False):
    df = df_raw.with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.month().alias("month"),
        pl.col("passenger_count").shift(1).alias("lag_1"),
        pl.col("passenger_count").shift(7).alias("lag_7"),
    # media semanal de 'passenger_count' con shift de 1 (dia actual excluido)    
    pl.col("passenger_count").shift(1).rolling_mean(window_size=7).alias("rolling_7_avg") 
    ])

    return df.drop_nulls()

def run_experiment(run_name, params, include_cyclical = False):
    df_features = create_features(df, include_cyclical)
    features = ["day_of_week", "month", "lag_1", "lag_7", "rolling_7_avg"]
    target = "passenger_count"

    split_idx = int(len(df_features) * 0.8) # split for time series scenario!
    train_df = df_features.slice(0, split_idx)
    val_df = df_features.slice(split_idx, len(df_features) - split_idx)

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    # training and logging
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "model_type": "XGBoost",
            "feature_set": "cyclical" if include_cyclical else "baseline_lags"
        })

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)

        rmse = np.sqrt(np.mean((y_val.to_numpy().ravel() - preds) ** 2))
        
        # logs 
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model"
        )
        print(f"Modelo registrado con éxito en MLflow! [{run_name}] RMSE: {rmse:.4f}")

# experiments
experiment_configs = [
    {"name": "xgb_baseline", "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, "cyclical": False},
    {"name": "xgb_more_trees", "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03}, "cyclical": False},
    {"name": "xgb_cyclical_features", "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03}, "cyclical": True},
]

for config in experiment_configs:
    run_experiment(config["name"], config["params"], config["cyclical"])
    