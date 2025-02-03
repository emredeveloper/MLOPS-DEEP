from datetime import datetime
from typing import Dict, Optional

def check_model_performance(current_metrics: Dict[str, float], threshold: float = 0.8) -> bool:
    """Check if model needs retraining based on performance metrics"""
    return current_metrics.get('accuracy', 0.0) < threshold

def trigger_retraining(data_path: Optional[str] = None) -> None:
    """Trigger automated model retraining"""
    from deepseek_mlops.train import train_model
    # Implementation for automated retraining
