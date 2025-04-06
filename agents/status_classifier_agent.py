
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

class ResolutionStatusClassifierAgent:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, data: pd.DataFrame):
        """
        Expects a DataFrame with columns: 'Issue Category', 'Priority', 'Sentiment', 'Resolution Status'
        """
        data = data.dropna(subset=['Issue Category', 'Priority', 'Sentiment', 'Resolution Status'])

        # Combine textual features
        data['combined'] = data['Issue Category'] + " " + data['Priority'] + " " + data['Sentiment']

        X = self.vectorizer.fit_transform(data['combined'])
        y = data['Resolution Status']

        self.model.fit(X, y)
        self.is_trained = True

        # Optional: Save model
        with open('models/status_classifier_model.pkl', 'wb') as f:
            pickle.dump((self.vectorizer, self.model), f)

    def load_model(self, model_path='models/status_classifier_model.pkl'):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.vectorizer, self.model = pickle.load(f)
                self.is_trained = True
        else:
            raise FileNotFoundError("Model file not found!")

    def predict_status(self, issue_category, priority, sentiment):
        if not self.is_trained:
            raise Exception("Model not trained or loaded!")

        combined = issue_category + " " + priority + " " + sentiment
        X = self.vectorizer.transform([combined])
        return self.model.predict(X)[0]
