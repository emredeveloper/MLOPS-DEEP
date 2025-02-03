import mlflow
import mlflow.sklearn

def start_run():
    """Starts an MLflow run."""
    mlflow.start_run()

def end_run():
    """Ends an MLflow run."""
    mlflow.end_run()  # Change to mlflow.end_run()

def log_params(params: dict):
    """Logs parameters to MLflow."""
    mlflow.log_params(params)  # Log all params at once

def log_metrics(metrics: dict):
    """Logs metrics to MLflow."""
    mlflow.log_metrics(metrics)  # Log all metrics at once

def log_model(model, model_name="model"):
    """Logs a model to MLflow."""
    mlflow.sklearn.log_model(model, model_name)