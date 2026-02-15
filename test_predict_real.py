
import requests
import os

ENDPOINT = "http://127.0.0.1:8000/predict"
TEST_FILE = r"C:\Users\Rishi\Downloads\Dataset\ch_b1\chb01_03.edf"

def test_prediction():
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        return

    print(f"Testing prediction with file: {TEST_FILE}")
    
    try:
        with open(TEST_FILE, "rb") as f:
            files = {"file": (os.path.basename(TEST_FILE), f, "application/octet-stream")}
            response = requests.post(ENDPOINT, files=files)
            
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Error Response:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_prediction()
