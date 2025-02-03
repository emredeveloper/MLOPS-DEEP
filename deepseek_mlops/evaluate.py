from sklearn.metrics import accuracy_score
from deepseek_mlops.utils import log

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    log(f"[INFO] Model evaluated. Accuracy: {accuracy:.4f}")
    return accuracy # Return accuracy