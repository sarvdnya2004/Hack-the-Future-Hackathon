import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

class PriorityPredictorAgent:
    def __init__(self):
        self.model = None
        self.vectorizer = None

    def train(self, df):
        """
        Train the priority prediction model.

        Args:
            df (pd.DataFrame): DataFrame with 'Issue Category', 'Sentiment', and 'Priority' columns.
        """
        df['combined_text'] = df['Issue Category'] + " " + df['Sentiment']
        X = df['combined_text']
        y = df['Priority']
        
        pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
        pipeline.fit(X, y)
        self.model = pipeline

    def predict_priority(self, issue_category, sentiment):
        """
        Predict priority based on issue and sentiment.

        Args:
            issue_category (str): The category of the issue.
            sentiment (str): Sentiment label.

        Returns:
            str: Predicted priority level.
        """
        input_text = f"{issue_category} {sentiment}"
        return self.model.predict([input_text])[0]

    def save_model(self, path='priority_model.pkl'):
        if self.model:
            joblib.dump(self.model, path)

    def load_model(self, path='priority_model.pkl'):
        self.model = joblib.load(path)
