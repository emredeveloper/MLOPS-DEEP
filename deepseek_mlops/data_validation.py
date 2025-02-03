import pandas as pd
from typing import Optional
from deepseek_mlops.utils import log

def validate_data(data: pd.DataFrame, threshold: Optional[float] = None) -> bool:
    """
    Validates the input data.
    Args:
        data: Input DataFrame
        threshold: Optional validation threshold
    Returns:
        bool: True if validation passes, False otherwise
    """
    if data is None or data.empty:
        log("[ERROR] Data is empty or None")
        return False

    if data.isnull().sum().sum() > 0:
        log("[ERROR] Data contains missing values")
        return False

    if not all(data.dtypes == 'float64'):
        log("[ERROR] Data contains non-numeric values")
        return False

    log("[INFO] Data validation passed")
    return True
