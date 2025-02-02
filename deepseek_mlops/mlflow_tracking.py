import mlflow
import mlflow.sklearn

def start_run():
    """MLflow deneyi başlatır."""
    mlflow.start_run()

def end_run():
    """MLflow deneyi sonlandırır."""
    mlflow.end_run()

def log_params(params: dict):
    """Parametreleri MLflow'a kaydeder."""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    """Metrikleri MLflow'a kaydeder."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model(model, model_name="model"):
    """Modeli MLflow'a kaydeder."""
    mlflow.sklearn.log_model(model, model_name)
