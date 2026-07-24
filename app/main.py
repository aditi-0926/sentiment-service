from fastapi import FastAPI
from app.schemas import TextIn, PredictionOut
from app.model import predict

app = FastAPI(title="Sentiment Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(payload: TextIn):
    label, score = predict(payload.text)
    return {"label": label, "confidence": score}