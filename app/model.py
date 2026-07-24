from transformers import pipeline

MODEL_PATH = "models/final"
LABELS = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

clf = pipeline("text-classification", model=MODEL_PATH, tokenizer=MODEL_PATH, top_k=1)

def predict(text: str):
    result = clf(text)[0][0]
    label = LABELS.get(result["label"], result["label"])
    return label, round(float(result["score"]), 4)