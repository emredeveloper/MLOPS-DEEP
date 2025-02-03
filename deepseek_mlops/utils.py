import logging
import time
import os
from deepseek_mlops.config import LOGS_DIR, LOG_FILE

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logging(log_file):
    """Sets up logging to a file."""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",  # Include log level
                        encoding="utf-8")

def log(message, level=logging.INFO):  # Add log level parameter
    """Logs messages to the file and console."""
    logging.log(level, message) # Use logging module for logging
    print(message) # Print to console as well


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        log(f"[TIMER] {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper