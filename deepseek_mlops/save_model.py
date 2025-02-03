# deepseek_mlops/save_model.py
import joblib
import os  # Import os for path manipulation
from deepseek_mlops.config import MODELS_DIR, TRAIN_TEST_SPLIT, RANDOM_STATE, N_ESTIMATORS  # Import config values

def save_trained_model(model, model_filename="trained_model.pkl"):  # Allow custom model filename
    """Saves a trained machine learning model to a file."""

    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True) # create the directory if it doesn't exist

    model_path = os.path.join(MODELS_DIR, model_filename)  # Construct full path
    joblib.dump(model, model_path)  # Save the model
    print(f"Trained model saved to {model_path}")


# Example usage (you would call this function from your training script)
# from sklearn.ensemble import RandomForestClassifier  # Example model
# model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
# # ... (train your model) ...
# save_trained_model(model)  # Save the trained model