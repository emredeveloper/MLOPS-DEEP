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
from deepseek_mlops.utils import log
import logging
from sklearn.ensemble import RandomForestClassifier  # Ensure RandomForestClassifier is imported

app = FastAPI(title="DeepSeek MLOps API")

# Jinja2 for HTML templates
templates = Jinja2Templates(directory="templates")  # Create a 'templates' directory

# Add prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Time for predictions')

# Model loading with proper error handling
try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found at: {MODEL_FILE}")
    
    model = joblib.load(MODEL_FILE)
    
    # Ensure we have the correct model type
    if not isinstance(model, RandomForestClassifier):
        log("[ERROR] Loaded model is not a RandomForestClassifier")
        model = None
    else:
        log(f"[INFO] RandomForestClassifier loaded successfully from: {MODEL_FILE}")
except Exception as e:
    log(f"[ERROR] Failed to load model: {str(e)}", level=logging.ERROR)
    model = None

class PredictionInput(BaseModel):
    features: list[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]  # Example Iris features
            }
        }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Make prediction with the model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    try:
        # Validate input length
        if len(input_data.features) != 4:  # Iris dataset has 4 features
            raise HTTPException(
                status_code=400, 
                detail="Input must have exactly 4 features"
            )
        
        # Make prediction
        features = np.array(input_data.features).reshape(1, -1)
        prediction = model.predict(features)
        
        # Track metrics
        PREDICTION_COUNTER.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return {
            "prediction": int(prediction[0]),
            "prediction_time": f"{(time.time() - start_time):.4f} seconds"
        }
    
    except Exception as e:
        log(f"[ERROR] Prediction failed: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain():
    """Endpoint to trigger model retraining"""
    try:
        from deepseek_mlops.retraining import trigger_retraining
        success = trigger_retraining()
        if success:
            return {"status": "success", "message": "Model retrained successfully"}
        else:
            raise HTTPException(status_code=500, detail="Retraining failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/debug")
async def debug():
    """Debug endpoint to check model status"""
    return {
        "model_type": str(type(model)),
        "model_params": model.get_params() if model else None,
        "model_file_exists": os.path.exists(MODEL_FILE),
        "model_file_path": MODEL_FILE,
        "model_loaded": model is not None
    }