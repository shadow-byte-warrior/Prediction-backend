from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
import shutil
import os
import uvicorn
from contextlib import asynccontextmanager
from preprocessing import preprocess_edf
from model import build_seizure_model

# Global model variable
model = None
MODEL_PATH = "best_seizure_model.h5"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        print(f"Loading model architecture and weights from {MODEL_PATH}...")
        if os.path.exists(MODEL_PATH):
            # Build the model architecture
            # Input shape must match what was used during training (1024, 18)
            model = build_seizure_model((1024, 18))
            
            # Load the weights
            model.load_weights(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
    yield
    # Clean up resources if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Seizure Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Save uploaded file temporarily
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess the file
        # This returns shape (n_windows, 1024, 18) or similar
        segments = preprocess_edf(temp_filename)
        
        if segments is None or len(segments) == 0:
            raise HTTPException(status_code=400, detail="Could not process EDF file or file is too short")
            
        # Run inference
        # Model output is probability (sigmoid) usually, or softmax
        predictions = model.predict(segments)
        
        # Calculate statistics
        # Assuming binary classification where >0.5 is seizure
        mean_prob = float(np.mean(predictions))
        max_prob = float(np.max(predictions))
        
        is_seizure = mean_prob > 0.5
        
        return {
            "filename": file.filename,
            "seizure_detected": bool(is_seizure),
            "average_probability": mean_prob,
            "max_probability": max_prob
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
