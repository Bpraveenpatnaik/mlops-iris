from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

import ml_utils


@asynccontextmanager
async def lifespan(app: FastAPI):
    # train the candidate models on startup and keep the best one loaded
    ml_utils.load_model()
    yield


app = FastAPI(title="Iris Predictor", docs_url="/", lifespan=lifespan)

FlowerClass = Literal["Iris Setosa", "Iris Versicolour", "Iris Virginica"]


# payload for a prediction: all measurements in cm, must be positive
class QueryIn(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# prediction response, with the model used and the time of the prediction (UTC)
class QueryOut(BaseModel):
    flower_class: FlowerClass
    model: str
    timestamp: datetime


# payload for the feedback loop: measurements plus the correct class
class FeedbackIn(QueryIn):
    flower_class: FlowerClass


@app.get("/ping")
def ping():
    """Health check."""
    return {"ping": "pong"}


@app.post("/predict_flower", response_model=QueryOut, status_code=200)
def predict_flower(query_data: QueryIn):
    """Predict the Iris species from the flower's measurements."""
    return {
        "flower_class": ml_utils.predict(query_data),
        "model": ml_utils.model_name,
        "timestamp": datetime.now(timezone.utc),
    }


@app.post("/feedback_loop", status_code=200)
def feedback_loop(data: List[FeedbackIn]):
    """Retrain the model with labelled samples from users."""
    ml_utils.retrain(data)
    return {"detail": "Feedback loop successful", "samples_received": len(data)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
