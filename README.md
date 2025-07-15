# Stock Predictor API

An end-to-end stock price prediction engine using XGBoost, technical indicators, and FastAPI. Designed to emulate real-world engineering.

## Features
- Multi-horizon price prediction (1, 5, 10 days)
- 30+ technical indicators (RSI, MACD, SMA, etc.)
- Model training using XGBoost
- API-ready with FastAPI and Docker

## Project Structure
- `data_pipeline.py`: Downloads and engineers features
- `model_pipeline.py`: Trains models and saves `.pkl` files
- `backend/`: FastAPI + Docker-based inference server
  - `main.py`: FastAPI app
  - `predictor.py`: Prediction logic
  - `data/processed/`: CSV data (mounted inside Docker)
- `models/`: Saved models (ignored in Git)

## Setup
```bash
pip install -r requirements.txt
python data_pipeline.py
python model_pipeline.py

# From the project root
docker build -t stock-predictor-backend ./backend

# Run the container and mount the data volume
docker run -p 8000:8000 -v "$(pwd)/backend/data/processed:/app/data/processed" stock-predictor-backend