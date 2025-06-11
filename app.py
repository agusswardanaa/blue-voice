from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.inference import predict_sentiment

app = FastAPI()

origins = [
    "http://127.0.0.1:5500"
]

# Tambahkan middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class SentimentRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to Blue Voice API!", "version": "1.0", "documentation": "/docs"}

# Predict endpoint
@app.post("/predict")
def predict(request: SentimentRequest):
    """
    Endpoint to predict sentiment from the input text.
    """

    result = predict_sentiment(request.text)
    return {"status": "success", "data": result}