import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost

# explicit local DB for mlflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flight_demand_forecasting_example")

# mocked data with dates and passenger counts + feature eng
dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="D")
df = pd.DataFrame({
    "date": dates,
    "passenger_count": np.random.randint(100, 300, size=len(dates))
}).sort_values("date")

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["lag_1"] = df["passenger_count"].shift(1)
df["lag_7"] = df["passenger_count"].shift(7)
df["rolling_7_avg"] = df["passenger_count"].shift(1).rolling(7).mean()
df = df.dropna()

# split for time series scenario!
features = ["day_of_week", "month", "lag_1", "lag_7", "rolling_7_avg"]
target = "passenger_count"

split_idx = int(len(df) * 0.8)
X_train, y_train = df[features].iloc[:split_idx], df[target].iloc[:split_idx]
X_val, y_val = df[features].iloc[split_idx:], df[target].iloc[split_idx:]

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
    rmse = np.sqrt(np.mean((y_val - preds) ** 2))
    
    # logs 
    mlflow.log_params(params)
    mlflow.log_metric("rmse", rmse)
    mlflow.xgboost.log_model(
        xgb_model=model,
        name="model"
    )
    print(f"Modelo registrado con éxito en MLflow! RMSE: {rmse:.4f}")
    