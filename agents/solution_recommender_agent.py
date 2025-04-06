# agents/solution_recommender_agent.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

class SolutionRecommenderAgent:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.vectorizer = TfidfVectorizer()
        self.trained = False

    def preprocess(self, df):
        df["Combined_Features"] = df["Issue Category"] + " " + df["Priority"] + " " + df["Sentiment"]
        return df

    def train(self, df):
        df = self.preprocess(df)
        X = self.vectorizer.fit_transform(df["Combined_Features"])
        y = df["Solution"]
        self.model.fit(X, y)
        self.trained = True

    def recommend_solution(self, issue_category, priority, sentiment):
        if not self.trained:
            raise ValueError("Model not trained. Call train() before prediction.")

        input_text = issue_category + " " + priority + " " + sentiment
        X_input = self.vectorizer.transform([input_text])
        return self.model.predict(X_input)[0]
