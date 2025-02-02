from sklearn.ensemble import RandomForestClassifier
from deepseek_mlops.config import N_ESTIMATORS, RANDOM_STATE
from deepseek_mlops.utils import log, measure_time
from deepseek_mlops.mlflow_tracking import log_params

@measure_time
def train_model(X_train, y_train):
    """Rastgele orman modeli ile eğitimi gerçekleştirir."""
    params = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE}
    log_params(params)  # Parametreleri MLflow'a kaydet

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    log("[INFO] Model başarıyla eğitildi.")
    return model