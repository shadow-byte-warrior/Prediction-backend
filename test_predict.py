
import requests
import os

# Adjust this path based on where we found the file
EDF_PATH = r"C:\Users\Rishi\Downloads\Dataset\ch_b1\chb01_03.edf"
URL = "http://127.0.0.1:8000/predict"

if not os.path.exists(EDF_PATH):
    print(f"File not found: {EDF_PATH}")
else:
    print(f"Testing prediction with {EDF_PATH}...")
    try:
        with open(EDF_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(URL, files=files)
            
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
        
    except Exception as e:
        print(f"Request failed: {e}")
