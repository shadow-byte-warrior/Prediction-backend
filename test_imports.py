
try:
    print("Importing tensorflow...")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    print("Importing fastapi...")
    from fastapi import FastAPI
    print("FastAPI imported.")
    
    print("Importing uvicorn...")
    import uvicorn
    print("Uvicorn imported.")
    
    print("Importing preprocessing...")
    from preprocessing import preprocess_edf
    print("Preprocessing imported.")
    
    print("Importing numpy...")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    print("Importing mne...")
    import mne
    print(f"MNE version: {mne.__version__}")
    
    print("All imports successful.")
except Exception as e:
    print(f"Error importing: {e}")
    import traceback
    traceback.print_exc()
