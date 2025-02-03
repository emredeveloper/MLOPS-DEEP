from sklearn.model_selection import train_test_split
from deepseek_mlops.config import TRAIN_TEST_SPLIT, RANDOM_STATE

def preprocess(data):
    """Splits the data into train and test sets."""
    X = data.drop(columns=["target"])
    y = data["target"]
    return train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)