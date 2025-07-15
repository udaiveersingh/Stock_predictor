# model_pipeline.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from typing import List

DATA_DIR = r"C:\Users\LENOVO\OneDrive\Documents\End-End Stock Prediction\Stock_predictor\backend\data\processed"
MODEL_DIR = r"C:\Users\LENOVO\OneDrive\Documents\End-End Stock Prediction\Stock_predictor\backend\models"
HORIZONS = [1, 5, 10]  # Forecast 1, 5, and 10 days ahead

def create_targets(df: pd.DataFrame, target_col=None, horizons=HORIZONS):
    if not target_col:
        # Auto-select the column that starts with 'Close_'
        target_candidates = [col for col in df.columns if col.startswith("Close_")]
        if not target_candidates:
            raise ValueError("No Close_ column found!")
        target_col = target_candidates[0]
    for h in horizons:
        df[f"target_t+{h}"] = df[target_col].shift(-h)
    df.dropna(inplace=True)
    return df

def train_model(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)
    return model

def process_and_train(ticker: str):
    print(f"\nTraining models for {ticker}")
    df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv")

    # Drop non-numeric/non-feature columns
    df = create_targets(df)
    df = df.select_dtypes(include='number')

    feature_cols = [col for col in df.columns if not col.startswith("target")]
    X = df[feature_cols]

    os.makedirs(MODEL_DIR, exist_ok=True)

    for h in HORIZONS:
        y = df[f"target_t+{h}"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = train_model(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"t+{h} | MSE: {mse:.4f}")

        dump(model, f"{MODEL_DIR}/{ticker}_t+{h}.pkl")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    for ticker in tickers:
        process_and_train(ticker)