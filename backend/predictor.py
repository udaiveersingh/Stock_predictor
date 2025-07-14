import joblib
import pandas as pd
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(ticker: str, horizon: int):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_t+{horizon}.pkl")
    model = joblib.load(model_path)
    return model

def predict(ticker: str, horizon: int, recent_data: pd.DataFrame):
    model = load_model(ticker, horizon)
    features = recent_data.tail(1).drop(columns=["Date", "Ticker"])  # Latest row, drop unused
    prediction = model.predict(features)[0]
    return prediction