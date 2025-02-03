import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import io  # Added for StringIO
from deepseek_mlops.config import MODEL_FILE
from pydantic import BaseModel
import numpy as np
from prometheus_client import make_asgi_app, Counter, Histogram
import time

app = FastAPI(title="DeepSeek MLOps API")

# Jinja2 for HTML templates
templates = Jinja2Templates(directory="templates")  # Create a 'templates' directory

# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time for predictions')

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

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    start_time = time.time()
    try:
        prediction = model.predict([input_data.features])
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}