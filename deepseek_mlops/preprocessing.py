from sklearn.model_selection import train_test_split
from deepseek_mlops.config import TRAIN_TEST_SPLIT, RANDOM_STATE

def preprocess(data, target_column="target"):
    """Splits the data into train and test sets."""
    if target_column not in data.columns:
        raise ValueError(f"Column '{target_column}' not found in data.")
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)