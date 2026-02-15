
import h5py
import tensorflow as tf

MODEL_PATH = "best_seizure_model.h5"

print(f"Inspecting {MODEL_PATH} with h5py...")
try:
    with h5py.File(MODEL_PATH, 'r') as f:
        print("Keys:", list(f.keys()))
        if 'model_config' in f.attrs:
            print("Found model_config attribute.")
            print(f.attrs['model_config'])
        else:
            print("No model_config attribute found.")
            
        if 'layer_names' in f.attrs:
            print("Layer names:", f.attrs['layer_names'])
            
except Exception as e:
    print(f"Error reading H5: {e}")

print("\nAttrib: Trying to load with compile=False...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Success loading with compile=False!")
    model.summary()
except Exception as e:
    print(f"Failed with compile=False: {e}")
    import traceback
    traceback.print_exc()
