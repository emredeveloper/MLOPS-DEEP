import pandas as pd
from deepseek_mlops.utils import log

def validate_data(data: pd.DataFrame) -> bool:
    """Validates the input data."""
    if data.isnull().sum().sum() > 0:
        log("[ERROR] Data contains missing values.")
        return False
    if not all(data.dtypes == 'float64'):
        log("[ERROR] Data contains non-numeric values.")
        return False
    log("[INFO] Data validation passed.")
    return True
