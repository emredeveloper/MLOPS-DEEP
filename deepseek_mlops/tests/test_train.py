import pytest
from deepseek_mlops.train import train_model
from sklearn.datasets import make_classification

def test_train_model():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")
