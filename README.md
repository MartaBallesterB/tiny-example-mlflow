# tiny-example-mlflow

*Scope:* end-to-end time series pipeline to forecast flight passenger demand.

*Data:* mocked dataset with dates, routes, and passenger counts simulating airline traffic. 

*Feature engineering:* vectorized time-based features (1 and 7 day lags + 7 day rolling average).

*Temporal Split:* strict chronological 80/20 train/val split (data leakage prevention).

*MLflow Tracking:* log hyperparameters, RMSE metric and the binary model artifact directly into mlflow.db (SQLite).


MVP: tracking experiments with MLFlow, comparing different time-series models + hyperparameter tuning.