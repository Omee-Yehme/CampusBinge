from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model.sentiment import predict as model_predict
from pydantic import BaseModel
from typing import List


MODEL_NAME = "sentiment_model"

app = FastAPI(title="Career ML Sentiment Service")


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1,
    description="Input text for sentiment analysis")


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model: str


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = model_predict(request.text)
    return {
        "label": result["label"],
        "confidence": result["confidence"],
        "model": MODEL_NAME
    }

class BatchPredictionRequest(BaseModel):
    texts: List[str]


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    if not request.texts:
        raise HTTPException(
            status_code=422, detail="Input list cannot be empty")

    results = []

    for text in request.texts:
        result = model_predict(text)
        results.append(result)

    return {
        "model": MODEL_NAME,
        "predictions": results
    }
