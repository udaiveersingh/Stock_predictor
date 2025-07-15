from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import logging
from predictor import predict

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define where your CSV data is
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

app = FastAPI()

class PredictionRequest(BaseModel):
    ticker: str
    horizon: int  # e.g., 1, 5, 10

@app.post("/predict")
def predict_price(req: PredictionRequest):
    csv_path = os.path.join(DATA_DIR, f"{req.ticker}.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Data not found")

    try:
        df = pd.read_csv(csv_path)
        prediction = predict(req.ticker, req.horizon, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    logging.info(f"Prediction for {req.ticker} at horizon {req.horizon}: {prediction}")

    return {
        "ticker": req.ticker,
        "horizon": req.horizon,
        "prediction": round(float(prediction), 2)
    }