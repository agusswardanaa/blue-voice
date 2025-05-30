from fastapi import FastAPI
import pandas as pd

from src.inference import predict_sentiment

app = FastAPI()

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Blue Voice API!", "version": "1.0", "documentation": "/docs"}

# Predict endpoint
@app.post("/predict")
def predict(text: str):
    """
    Endpoint to predict sentiment from the input text.
    """

    result = predict_sentiment(text)
    return {"status": "success", "data": result}