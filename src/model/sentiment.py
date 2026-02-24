from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME
)

def predict(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    text = text[:512]  # truncate long input

    result = _classifier(text)[0]

    return {
        "label": result["label"],
        "confidence": float(result["score"])
    }