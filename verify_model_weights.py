
import os
import tensorflow as tf
from model import build_seizure_model
import sys

# Force UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

MODEL_PATH = "best_seizure_model.h5"

def verify_weights():
    print(f"TensorFlow Version: {tf.__version__}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    try:
        # Build the model
        input_shape = (1024, 18) 
        print(f"Building model with input shape: {input_shape}")
        model = build_seizure_model(input_shape)
        # model.summary() # Removed summary to avoid console width issues
        
        # Load weights
        print(f"Attempting to load weights from {MODEL_PATH}...")
        model.load_weights(MODEL_PATH)
        print("SUCCESS: Weights loaded successfully!")
        
        # Verify prediction
        import numpy as np
        dummy_input = np.random.random((1, 1024, 18))
        prediction = model.predict(dummy_input)
        print(f"Test prediction successful. Output shape: {prediction.shape}, Value: {prediction[0][0]}")
        
    except Exception as e:
        print(f"FAILED to load weights: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_weights()
