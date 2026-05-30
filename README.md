<<<<<<< HEAD
# ReviewSense

ReviewSense is an end-to-end machine learning project for movie review sentiment analysis.

The project trains a sentiment classification model on a Turkish review dataset and serves predictions through a FastAPI REST API.

## Features

- Movie review sentiment classification
- Text cleaning and preprocessing
- HTML tag removal
- TF-IDF vectorization
- Logistic Regression classifier
- Model evaluation with accuracy, precision, recall and F1-score
- Saved model with Joblib
- REST API with FastAPI
- Local prediction script
- Interactive Swagger documentation

## Dataset

The repository includes `data/reviews.csv` add your dataset`.

## Installation

Create virtual environment:

```powershell
python -m venv venv
```

Activate virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Train the Model

```powershell
python src/train.py
```

This command:

- loads the dataset from `data/reviews.csv` (or falls back to `data/imdb_reviews.csv` if present)
- cleans review texts
- trains the TF-IDF + Logistic Regression model
- evaluates model performance
- saves the model to `models/sentiment_model.pkl`

## Run Local Prediction

```powershell
python src/predict.py
```

Example output:

```json
{
  "review": "This movie was amazing. The story and acting were excellent.",
  "sentiment": "positive",
  "confidence": 0.9015,
  "probabilities": {
    "negative": 0.0985,
    "positive": 0.9015
  }
}
```

## Run the API

```powershell
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## API Usage

Endpoint:

```http
POST /predict
```

Request:

```json
{
  "review": "This movie was boring and terrible."
}
```

Response:

```json
{
  "sentiment": "negative",
  "confidence": 0.9947,
  "probabilities": {
    "negative": 0.9947,
    "positive": 0.0053
  }
}
```

## How It Works

ReviewSense uses a classic machine learning pipeline:

```text
Raw review text
↓
Text cleaning
↓
TF-IDF vectorization
↓
Logistic Regression classifier
↓
Sentiment prediction
```

## Technologies

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Pydantic

## Future Improvements

- Add neutral sentiment class
- Add Docker support
- Add automated tests
- Add web interface
- Deploy API
- Integrate with a movie review website

## License

This project is for educational and portfolio purposes.
