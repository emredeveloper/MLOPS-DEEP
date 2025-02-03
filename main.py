import mlflow
import logging  # Add logging import
from deepseek_mlops.data_loader import load_data
from deepseek_mlops.preprocessing import preprocess  # Added import for preprocessing
from deepseek_mlops.train import train_model
from deepseek_mlops.evaluate import evaluate_model
from deepseek_mlops.save_model import save_trained_model
from deepseek_mlops.utils import log, setup_logging
from deepseek_mlops.config import MODEL_FILE, LOG_FILE  # Add LOG_FILE import

def main():
    setup_logging(LOG_FILE)  # Set up logging at the start of your application
    log("[INFO] Model training process started.")

    mlflow.set_experiment("Deepseek_MLOps_Experiment")  # Set your MLflow experiment
    with mlflow.start_run():
        # 1. Load Data
        data = load_data()
        if data is None:  # Handle data loading errors
            log("[ERROR] Data loading failed. Exiting.", level=logging.ERROR)
            return  # Exit the program

        # 2. Preprocess Data
        X_train, X_test, y_train, y_test = preprocess(data)
        if X_train is None:  # Handle preprocessing errors
            log("[ERROR] Preprocessing failed. Exiting.", level=logging.ERROR)
            return

        # 3. Train Model
        model = train_model(X_train, y_train)

        # 4. Evaluate Model
        accuracy = evaluate_model(model, X_test, y_test)  # Get the returned accuracy

        # 5. Log Metrics to MLflow
        mlflow.log_metric("accuracy", accuracy) # log the accuracy

        # 6. Save Model (locally and to MLflow)
        save_trained_model(model)  # Save locally
        mlflow.sklearn.log_model(model, "model")  # Save to MLflow

        log("[INFO] Model training process completed successfully.")

if __name__ == "__main__":
    main()