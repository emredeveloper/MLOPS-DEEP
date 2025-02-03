import mlflow
from typing import Optional, Union

def register_model(model_name: str, run_id: Optional[str] = None) -> Union[int, str]:
    """Register model to MLflow Model Registry"""
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    return result.version
