from sklearn.ensemble import RandomForestClassifier
from deepseek_mlops.config import N_ESTIMATORS, RANDOM_STATE
from deepseek_mlops.utils import log, measure_time
from deepseek_mlops.mlflow_tracking import log_params

@measure_time
def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model."""
    params = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE}
    log_params(params)

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    log("[INFO] Model trained successfully.")
    return model