# Stock Predictor API

An end-to-end stock price prediction engine using XGBoost, technical indicators, and FastAPI. Designed to emulate real-world engineering.

## Features
- Multi-horizon price prediction (1, 5, 10 days)
- 30+ technical indicators (RSI, MACD, SMA, etc.)
- Model training using XGBoost
- API-ready with FastAPI and Docker (coming next)

## Project Structure
- `data_pipeline.py`: Downloads and engineers features
- `model_pipeline.py`: Trains models and saves `.pkl` files
- `backend/`: FastAPI inference server (to be added)
- `models/`: Saved models (ignored in Git)
- `data/processed/`: CSV data (ignored in Git)

## Setup
```bash
pip install -r requirements.txt
python data_pipeline.py
python model_pipeline.py