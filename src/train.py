import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import html

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
FALLBACK_DATA_PATH = BASE_DIR / "data" / "imdb_reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"

MODEL_DIR.mkdir(exist_ok=True)


def print_word_weights(model):
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]

    words = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]

    word_weights = list(zip(words, weights))

    positive_words = sorted(word_weights, key=lambda x: x[1], reverse=True)[:20]
    negative_words = sorted(word_weights, key=lambda x: x[1])[:20]

    print("\nEn Pozitif Kelimeler:")
    for word, weight in positive_words:
        print(f"{word}: {weight:.3f}")

    print("\nEn Negatif Kelimeler:")
    for word, weight in negative_words:
        print(f"{word}: {weight:.3f}")


def test_samples(model):
    samples = [
        "This movie was fantastic! I loved every moment of it.",
        "The plot was terrible and the acting was worse. Don't waste your time.",
        "An average film with some good and some bad moments.",
        "I was on the edge of my seat the entire time. Highly recommend!",
        "The movie was a complete disappointment. I expected much more."
    ]

    print("\nÖrnek Tahminler:")
    for sample in samples:
        prediction = model.predict([sample])[0]
        probabilities = model.predict_proba([sample])[0]

        classes = model.classes_
        confidence = max(probabilities)

        print("-" * 50)
        print(f"Yorum: {sample}")
        print(f"Tahmin: {prediction}")
        print(f"Güven: {confidence:.2f}")

        for label, prob in zip(classes, probabilities):
            print(f"{label}: {prob:.2f}")


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def main():
    if DATA_PATH.exists():
        data_file = DATA_PATH
    elif FALLBACK_DATA_PATH.exists():
        data_file = FALLBACK_DATA_PATH
    else:
        raise FileNotFoundError(
            "Dataset not found. Please add data/reviews.csv or data/imdb_reviews.csv"
        )

    df = pd.read_csv(data_file)

    df = df.dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].apply(clean_text)

    print("\nTemizlenmiş Örnekler:")
    print(df["review"].head(3))

    print("\nVeri Dağılımı:")
    print(df["sentiment"].value_counts())

    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    

    

    model = Pipeline([
        ("vectorizer", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1
    )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nModel Performansı:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    print_word_weights(model)
    test_samples(model)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()