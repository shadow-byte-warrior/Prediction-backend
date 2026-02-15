
import tensorflow as tf
import os
import numpy as np

MODEL_PATH = "best_seizure_model.h5"

if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model Type: {type(model)}")
        
        try:
            print(f"Model Input Shape: {model.input_shape}")
            print(f"Model Output Shape: {model.output_shape}")
            model.summary()
        except Exception as e:
            print(f"Error inspecting shape: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Model not found at {MODEL_PATH}")
