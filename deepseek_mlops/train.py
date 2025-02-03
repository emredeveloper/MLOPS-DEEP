from sklearn.ensemble import RandomForestClassifier
from deepseek_mlops.config import N_ESTIMATORS, RANDOM_STATE
from deepseek_mlops.utils import log, measure_time

@measure_time
def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model."""
    try:
        # Create a basic RandomForestClassifier without any special parameters
        model = RandomForestClassifier(
            n_estimators=10,  # Simplified parameters
            random_state=42
        )
        model.fit(X_train, y_train)
        log("[INFO] RandomForestClassifier trained successfully.")
        return model
    except Exception as e:
        log(f"[ERROR] Model training failed: {str(e)}")
        raise