# tiny-example-mlflow

*Scope:* end-to-end time series pipeline to forecast flight passenger demand.
*Data:* mocked dataset with dates, routes, and passenger counts simulating airline traffic. 
*Temporal Split:* strict chronological 80/20 train/val split (data leakage prevention).
*MLflow Tracking:* log hyperparameters, RMSE metric and the binary model artifact directly into mlflow.db (SQLite).
