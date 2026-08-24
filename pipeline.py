import numpy as np
import polars as pl
#import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

# explicit local DB for mlflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flight_demand_forecasting_example")

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

df = df.with_columns([
    pl.col("date").dt.weekday().alias("day_of_week"),
    pl.col("date").dt.month().alias("month"),
    pl.col("passenger_count").shift(1).alias("lag_1"),
    pl.col("passenger_count").shift(7).alias("lag_7"),
    pl.col("passenger_count").shift(1).rolling_mean(window_size=7).alias("rolling_7_avg") # media semanal de 'passenger_count' con shift de 1 (dia actual excluido)
]).drop_nulls()

# split for time series scenario!
features = ["day_of_week", "month", "lag_1", "lag_7", "rolling_7_avg"]
target = "passenger_count"

split_idx = int(len(df) * 0.8)
train_df = df.slice(0, split_idx)
val_df = df.slice(split_idx, len(df) - split_idx)

X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]

# training and logging
with mlflow.start_run(run_name="single_xgboost_run"):
    params = {
        "n_estimators": 100, 
        "max_depth": 5, 
        "learning_rate": 0.05, 
        "random_state": 42
    }

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
    print(f"Modelo registrado con éxito en MLflow! RMSE: {rmse:.4f}")
    