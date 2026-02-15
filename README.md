
# Seizure Detection API

A FastAPI backend for seizure detection using a deep learning model (CNN-LSTM-Attention).

## Deployment

The application is deployed at:
`https://prediction-backend-nja4.onrender.com`

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Locally**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Body**: `file` (EDF file)

### Testing

Run the included verification scripts:

- **Local Test**:
  ```bash
  python test_predict_real.py
  ```

- **Remote Test**:
  ```bash
  python test_remote_deployment.py
  ```

## Project Structure
- `main.py`: FastAPI application.
- `model.py`: Model architecture definition.
- `preprocessing.py`: EEG data preprocessing.
- `render.yaml`: Render deployment configuration.
