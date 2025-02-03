from datetime import datetime
from typing import Optional, Dict, Any
import mlflow
from deepseek_mlops.data_loader import load_data
from deepseek_mlops.preprocessing import preprocess
from deepseek_mlops.train import train_model
from deepseek_mlops.evaluate import evaluate_model
from deepseek_mlops.save_model import save_trained_model
from deepseek_mlops.utils import log, measure_time

@measure_time
def check_model_performance(current_metrics: Dict[str, float], threshold: float = 0.8) -> bool:
    """Check if model needs retraining based on performance metrics"""
    accuracy = current_metrics.get('accuracy', 0)
    needs_retraining = accuracy < threshold
    log(f"[INFO] Current accuracy: {accuracy:.3f}, Threshold: {threshold}")
    return needs_retraining

@measure_time
def trigger_retraining(data_path: Optional[str] = None) -> bool:
    """
    Trigger automated model retraining
    Returns: bool indicating if retraining was successful
    """
    try:
        with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            data = load_data(data_path) if data_path else load_data()
            if data is None:
                return False

            X_train, X_test, y_train, y_test = preprocess(data)
            model = train_model(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            
            mlflow.log_metrics(metrics)
            
            if metrics['accuracy'] >= 0.8:
                save_trained_model(model)
                log("[INFO] Retraining successful - New model saved")
                return True
            
            log("[WARNING] Retraining yielded poor performance")
            return False

    except Exception as e:
        log(f"[ERROR] Retraining failed: {str(e)}")
        return False
