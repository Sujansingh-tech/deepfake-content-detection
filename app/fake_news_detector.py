import joblib
import os
import pandas as pd

model_path = os.path.join('model', 'fake_news_model.pkl')
vectorizer_path = os.path.join('model', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def detect_fake_news(text):
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)
    return "Fake News" if prediction[0] == 1 else "Real News"
