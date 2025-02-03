import logging  # Added import for logging
import pandas as pd
from sklearn.datasets import load_iris
from deepseek_mlops.utils import log

def load_data():
    """Loads the Iris dataset and returns it as a Pandas DataFrame."""
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        log("[INFO] Iris dataset loaded successfully.")
        return df
    except Exception as e:
        log(f"[ERROR] Error loading data: {e}", level=logging.ERROR)  # Log the error
        return None  # Or raise the exception if you want to stop execution