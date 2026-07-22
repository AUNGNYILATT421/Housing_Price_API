from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app import model as ml
from app.database import create_tables, get_db, save_prediction, get_predictions
from app.schemas import HouseFeatures, PredictionResponse, PredictionRecord


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    ml.load_artifacts()
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predicts house sale prices using a GBR + CatBoost + KernelRidge stacking ensemble trained on the Kaggle Housing Prices dataset (Top 16 / 4,000+ participants).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model": "loaded" if ml._stack is not None else "not loaded"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HouseFeatures, db: Session = Depends(get_db)):
    try:
        input_df = features.to_dataframe()
        price = ml.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    record = save_prediction(
        db=db,
        input_features=features.model_dump(by_alias=True),
        predicted_price=price,
    )

    return PredictionResponse(
        prediction_id=record.id,
        predicted_price=round(price, 2),
        predicted_price_formatted=f"${price:,.0f}",
    )


@app.get("/predictions", response_model=List[PredictionRecord], tags=["Prediction"])
def list_predictions(limit: int = 50, db: Session = Depends(get_db)):
    return get_predictions(db, limit=limit)
