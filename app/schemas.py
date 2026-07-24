from pydantic import BaseModel

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    confidence: float