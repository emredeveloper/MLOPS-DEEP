import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import io  # Added for StringIO
from deepseek_mlops.config import MODEL_FILE

app = FastAPI()

# Jinja2 for HTML templates
templates = Jinja2Templates(directory="templates")  # Create a 'templates' directory

# Load the trained model
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model loaded successfully from: {MODEL_FILE}")
except FileNotFoundError:
    print(f"Error: Model file not found at: {MODEL_FILE}")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, features: str = Form(...), csv_data: str = Form(None)):
    try:
        features_list = [float(x) for x in features.split(",")]
        prediction = model.predict([features_list])
        prediction_result = int(prediction)  # Store prediction result

        # Handle CSV data (if provided)
        if csv_data:
            try:
                df = pd.read_csv(io.StringIO(csv_data))  # Parse text as CSV
                #... (Process the CSV data as needed)...
                csv_message = "CSV data received and processed (example)."  # Placeholder
            except Exception as e:
                csv_message = f"Error processing CSV: {e}"
        else:
            csv_message = "No CSV data provided."

        return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction_result, "csv_message": csv_message})

    except ValueError:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Invalid input features. Please provide comma-separated numbers.", "csv_message": "No CSV data provided."})
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"An error occurred during prediction: {e}", "csv_message": "No CSV data provided."})