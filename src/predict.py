from pathlib import Path
import joblib
import html
import re

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_model():
    return joblib.load(MODEL_PATH)


def predict_sentiment(review: str):
    cleaned_review = clean_text(review)
    model = load_model()
    prediction = model.predict([cleaned_review])[0]
    probabilities = model.predict_proba([cleaned_review])[0]

    classes = model.classes_
    confidence = max(probabilities)

    return {
        "review": review,
        "sentiment": prediction,
        "confidence": round(float(confidence), 4),
        "probabilities": {
            label: round(float(prob), 4)
            for label, prob in zip(classes, probabilities)
        }
    }


if __name__ == "__main__":
    result = predict_sentiment(
        "This movie was amazing. The story and acting were excellent."
    )

    print(result)
