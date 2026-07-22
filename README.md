# House Price Prediction API

A production-ready REST API that predicts house sale prices using a stacking ensemble trained on the [Kaggle Housing Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Stack:** FastAPI · PostgreSQL · SQLAlchemy · scikit-learn · Docker · GCP Cloud Run

---

## Model

A stacking ensemble of three base learners with an SGDRegressor meta-model:

| Model | Role |
|---|---|
| GradientBoostingRegressor | Base learner |
| CatBoostRegressor | Base learner |
| KernelRidge (RBF) | Base learner |
| SGDRegressor | Meta-model |

Eight features are engineered at inference time: `TotalSF`, `TotalQualSF`, `Age`, `TimeSinceRemod`, `InteriorQualityScore`, `TotalBaths`, `HasPorchDeck`, `TotalPorchDeckSF`.

---

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI endpoints
│   ├── model.py         # Artifact loading and inference
│   ├── schemas.py       # Pydantic request/response models
│   └── database.py      # SQLAlchemy ORM and DB helpers
├── ml/
│   ├── preprocessing.py # Feature engineering pipeline
│   ├── export_model.py  # Train and export model artifacts
│   └── artifacts/       # Saved .joblib files (generated — not committed)
├── data/
│   └── train.csv        # Kaggle training data (not committed)
├── notebooks/
│   └── housing-prices-xgb-cat-krr.ipynb
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `train.csv` from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) placed in `data/`

### 1. Train the model

Run training inside Docker so the artifact versions match the API environment:

```bash
docker compose run --rm api python -m ml.export_model
```

This saves three artifacts to `ml/artifacts/`.

### 2. Start the API

```bash
docker compose up
```

- API: `http://localhost:8080`
- Interactive docs: `http://localhost:8080/docs`
- Database: PostgreSQL on port `5433`

### 3. Stop

```bash
docker compose down        # stop containers
docker compose down -v     # stop and delete database volume
```

---

## Running Without Docker

```bash
pip install -r requirements.txt

# Train (reads data/train.csv, saves ml/artifacts/)
python -m ml.export_model

# Start API (falls back to SQLite if DATABASE_URL is not set)
uvicorn app.main:app --reload --port 8080
```

---

## API Reference

### `GET /health`

Returns model load status.

```json
{"status": "ok", "model": "loaded"}
```

### `POST /predict`

Accepts house features and returns a predicted sale price. All fields have defaults — send only what you know.

**Request body (example):**

```json
{
  "GrLivArea": 1710,
  "TotalBsmtSF": 856,
  "OverallQual": 7,
  "YearBuilt": 2003,
  "YearRemodAdd": 2003,
  "GarageArea": 548,
  "Neighborhood": "CollgCr",
  "KitchenQual": "Gd",
  "ExterQual": "Gd"
}
```

**Response:**

```json
{
  "prediction_id": 1,
  "predicted_price": 213450.0,
  "predicted_price_formatted": "$213,450"
}
```

### `GET /predictions?limit=50`

Returns recent predictions from the database.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./predictions.db` | SQLAlchemy connection string |

Copy `.env.example` to `.env` and set values for local development.

---

## Deployment (GCP Cloud Run)

The Dockerfile exposes port `8080` (Cloud Run standard).

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/housing-price-api

# Deploy
gcloud run deploy housing-price-api \
  --image gcr.io/YOUR_PROJECT_ID/housing-price-api \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=postgresql://user:pass@host:5432/housing
```
