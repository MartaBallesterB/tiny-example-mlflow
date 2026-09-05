import numpy as np
import polars as pl
import xgboost as xgb
import mlflow

from dataclasses import dataclass
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit

# explicit local DB for mlflow tracking + xgboost autologging
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flight_demand_forecasting")
mlflow.xgboost.autolog(disable = True)

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
    """Split, training and tracking execution in MLFlow for XGBoost model. It includes cross-validation for time-series data."""

    def __init__(self, raw_data, target_col = "passenger_count", train_ratio = 0.8, n_splits = 3):
        self.raw_data = raw_data
        self.target_col = target_col
        self.train_ratio = train_ratio
        self.n_splits = n_splits

    def split_dev_test(self, df):
        split_idx = int(len(df) * self.train_ratio)
        # I'll use dev set to experiment with tuning (train + val)
        dev_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx)
        return dev_df, test_df

    def run(self, config: ExperimentConfig):

        engineer = FeatureEngineering()
        df_features, features = engineer.create_lag_and_rolling_features(self.raw_data) # lag creation before split for time-series to avoid Nulls in val set!

        dev_df, test_df = self.split_dev_test(df_features)
        ts_cross_val = TimeSeriesSplit(n_splits=self.n_splits)

        # Parent run:
        with mlflow.start_run(run_name=config.name) as parent_run:
            mlflow.set_tags({
                "model_type": "XGBoost",
                "feature_set": "baseline_lags",
                "cross_val_strategy": "TimeSeriesSplit",
                "n_splits": self.n_splits,
            })

            mlflow.log_params(config.params)
            mlflow.log_param("n_splits", self.n_splits)

            fold_rmse_list = []

            X_dev = dev_df[features].to_numpy()
            y_dev = dev_df[self.target_col].to_numpy()

            # Iterate over cv folds
            for fold, (train_idx, val_idx) in enumerate(ts_cross_val.split(X_dev)):
                X_train, X_val, y_train, y_val = X_dev[train_idx], X_dev[val_idx], y_dev[train_idx], y_dev[val_idx]
                
                # Child run for each fold + log in mflow!:
                with mlflow.start_run(run_name=f"fold_{fold + 1}", nested=True):
                    model = xgb.XGBRegressor(**config.params)
                    model.fit(X_train, y_train)

                    preds = model.predict(X_val)
                    rmse_fold = np.sqrt(np.mean((y_val - preds) ** 2))
                    fold_rmse_list.append(rmse_fold)

                    mlflow.log_metric("fold_rmse", rmse_fold)
                    mlflow.log_param("fold_number", fold + 1)

            cv_mean_rmse = float(np.mean(fold_rmse_list))
            cv_std_rmse = float(np.std(fold_rmse_list))

            mlflow.log_metric("cv_rmse_mean", cv_mean_rmse)
            mlflow.log_metric("cv_rmse_std", cv_std_rmse)

            print(f"[{config.name}] Completed cross-validation ({self.n_splits} folds). Mean RMSE: {cv_mean_rmse:.4f} (+/- std: {cv_std_rmse:.4f})")


if __name__ == "__main__":
    df_raw = TimeSeriesDataGenerator.generate_mock_data()
    runner = XGBoostTimeSeriesRunner(raw_data=df_raw, n_splits=3)

    experiments = [
        ExperimentConfig(name="xgb_baseline_cv3", params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}),
        ExperimentConfig(name="xgb_more_trees_cv3", params={"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03}),
        ExperimentConfig(name="xgb_deep_trees_cv3", params={"n_estimators": 100, "max_depth": 7, "learning_rate": 0.05}),
    ]

    for exp in experiments:
        runner.run(exp)


# 1) check internal optimization: loss function and regularization parameters. reg_alpha, reg_lambda
# 2) check external optimization: hyperparameter tuning with Optuna.

