import numpy as np
import polars as pl
import xgboost as xgb
import mlflow
import mlflow.xgboost

from dataclasses import dataclass
from typing import Dict, Any

# explicit local DB for mlflow tracking + xgboost autologging
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experiment_1_flight_demand_forecasting")
mlflow.xgboost.autolog(log_models=True)

class TimeSeriesDataGenerator:
    """Mocked data generator"""

    @staticmethod
    def generate_mock_data():
        dates = pl.date_range(
            start=pl.date(2025, 1, 1),
            end=pl.date(2025, 12, 31),
            interval="1d",
            eager=True
        )

        pl_df = pl.DataFrame({"date": dates, 
                              "passenger_count": pl.int_range(100, 300, eager=True).sample(n=len(dates), with_replacement=True)}).sort("date") # to keep the time series order!
        return pl_df

class FeatureEngineering:

    def create_lag_and_rolling_features(self, df_raw):
        features = ["day_of_week", "month", "lag_1", "lag_7", "rolling_7_avg"]

        df = df_raw.with_columns([
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.month().alias("month"),
            pl.col("passenger_count").shift(1).alias("lag_1"),
            pl.col("passenger_count").shift(7).alias("lag_7"),
            # media semanal de 'passenger_count' con shift de 1 (dia actual excluido)    
            pl.col("passenger_count").shift(1).rolling_mean(window_size=7).alias("rolling_7_avg") 
        ])

        return df, features

@dataclass
class ExperimentConfig:
    """Data container to define experiment parameters"""
    name: str
    params: Dict[str, Any]


class XGBoostTimeSeriesRunner:
    """Split, training and tracking execution in MLFlow for XGBoost model"""

    def __init__(self, raw_data, target_col = "passenger_count", train_ratio = 0.8):
        self.raw_data = raw_data
        self.target_col = target_col
        self.train_ratio = train_ratio

    def _split_data(self, df, features):
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.slice(0, split_idx)
        val_df = df.slice(split_idx)

        X_train, y_train = train_df[features], train_df[self.target_col]
        X_val, y_val = val_df[features], val_df[self.target_col]

        return X_train, y_train, X_val, y_val

    def run(self, config: ExperimentConfig):

        engineer = FeatureEngineering()
        df_features, features = engineer.create_lag_and_rolling_features(self.raw_data) # lag creation before split for time-series to avoid Nulls in val set!

        X_train, y_train, X_val, y_val = self._split_data(df_features, features)

        with mlflow.start_run(run_name=config.name):
            mlflow.set_tags({
                "model_type": "XGBoost",
                "feature_set": "baseline_lags"
            })

            model = xgb.XGBRegressor(**config.params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val.to_numpy().ravel() - preds) ** 2))

            print(f"[{config.name}] Successful execution! RMSE: {rmse:.4f}")


if __name__ == "__main__":
    df_raw = TimeSeriesDataGenerator.generate_mock_data()
    runner = XGBoostTimeSeriesRunner(raw_data=df_raw)

    experiments = [
        ExperimentConfig(name="xgb_baseline", params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}),
        ExperimentConfig(name="xgb_more_trees", params={"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03}),
        ExperimentConfig(name="xgb_deep_trees", params={"n_estimators": 100, "max_depth": 7, "learning_rate": 0.05}),
    ]

    for exp in experiments:
        runner.run(exp)


# 1) check internal optimization: loss function and regularization parameters. reg_alpha, reg_lambda
# 2) check external optimization: hyperparameter tuning with Optuna or Hyperopt.

