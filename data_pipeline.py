# data_pipeline.py

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from typing import List

def download_stock_data(ticker: str, period="2y", interval="1d") -> pd.DataFrame:
    # Download data and disable auto_adjust to prevent MultiIndex
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    # Flatten MultiIndex if it exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Add a few commonly used indicators
    df.ta.rsi(length=14, append=True)         # Relative Strength Index
    df.ta.macd(append=True)                   # MACD
    df.ta.sma(length=20, append=True)         # Simple Moving Average
    df.ta.ema(length=20, append=True)         # Exponential Moving Average
    df.ta.bbands(append=True)                 # Bollinger Bands
    df.ta.cci(append=True)                    # Commodity Channel Index
    df.ta.stoch(append=True)                  # Stochastic Oscillator
    df.ta.adx(append=True)                    # Average Directional Index
    df.ta.willr(append=True)                  # Williams %R

    df.dropna(inplace=True)
    return df

def prepare_data(tickers: List[str], save_path="data/processed/") -> None:
    os.makedirs(save_path, exist_ok=True)

    for ticker in tickers:
        try:
            print(f"\nProcessing {ticker}...")
            df = download_stock_data(ticker)
            df = add_technical_indicators(df)
            df.to_csv(f"{save_path}/{ticker}.csv", index=False)
            print(f"Saved to {save_path}/{ticker}.csv")
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]  # You can modify this list
    prepare_data(tickers)