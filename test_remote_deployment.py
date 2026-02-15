
import requests
import os
import sys

# Force UTF-8 encoding for stdout/stderr
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# PRODUCTION URL
ENDPOINT = "https://prediction-backend-nja4.onrender.com/predict"
TEST_FILE = r"C:\Users\Rishi\Downloads\Dataset\ch_b1\chb01_03.edf"

def test_remote_prediction():
    print(f"Testing REMOTE endpoint: {ENDPOINT}")
    
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        return
    
    try:
        print(f"Uploading file: {os.path.basename(TEST_FILE)}...")
        with open(TEST_FILE, "rb") as f:
            files = {"file": (os.path.basename(TEST_FILE), f, "application/octet-stream")}
            # Increase timeout cause cold starts on free tier can be slow
            response = requests.post(ENDPOINT, files=files, timeout=60)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error Response:", response.text)
            
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The server might be waking up (cold start). Try again.")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_remote_prediction()
