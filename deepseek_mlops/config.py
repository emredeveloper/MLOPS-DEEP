import os
import yaml

this_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(os.path.dirname(this_dir), "config.yaml")
with open(yaml_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

BASE_DIR = os.path.abspath(config_data["BASE_DIR"])
MODELS_DIR = os.path.abspath(config_data["MODELS_DIR"])
MODEL_FILE = os.path.abspath(config_data["MODEL_FILE"])
LOGS_DIR = os.path.abspath(config_data["LOGS_DIR"])
LOG_FILE = os.path.abspath(config_data["LOG_FILE"])
TRAIN_TEST_SPLIT = config_data["TRAIN_TEST_SPLIT"]
RANDOM_STATE = config_data["RANDOM_STATE"]
N_ESTIMATORS = config_data["N_ESTIMATORS"]