from typing import Dict, Any
import numpy as np
from prometheus_client import Counter, Histogram

# Metrics for monitoring
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time for predictions')
FEATURE_DRIFT = Histogram('feature_drift', 'Distribution drift in features')

def track_prediction_metrics(prediction: Any, latency: float, features: Dict) -> None:
    """Track various prediction metrics"""
    PREDICTION_COUNTER.inc()
    PREDICTION_LATENCY.observe(latency)
    for feature_name, value in features.items():
        FEATURE_DRIFT.labels(feature=feature_name).observe(value)
