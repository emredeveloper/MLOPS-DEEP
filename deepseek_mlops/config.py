import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/trained_model.pkl")
LOG_FILE = os.path.join(BASE_DIR, "../logs/training.log")

TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100  # RandomForest için