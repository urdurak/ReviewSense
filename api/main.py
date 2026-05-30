from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_sentiment

app = FastAPI(
    title="ReviewSense API",
    description="Movie review sentiment analysis API",
    version="1.0.0"
)


class ReviewRequest(BaseModel):
    review: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Movie review text"
    )


class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/")
def root():
    return {
        "message": "ReviewSense API is running"
    }


@app.post("/predict", response_model=ReviewResponse)
def predict(request: ReviewRequest):
    result = predict_sentiment(request.review)

    return {
        "sentiment": result["sentiment"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"]
    }
