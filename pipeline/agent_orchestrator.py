# pipeline/agent_orchestrator.py

import pandas as pd
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from agents.resolution_status_classifier_agent import ResolutionStatusClassifierAgent
from agents.solution_recommender_agent import SolutionRecommenderAgent
from agents.summarization_agent import SummarizationAgent

# Load dataset
df = pd.read_csv("data/ticket_data.csv")

# Initialize agents
sentiment_agent = SentimentAnalysisAgent()
status_agent = ResolutionStatusClassifierAgent()
solution_agent = SolutionRecommenderAgent()
summarizer_agent = SummarizationAgent()

# Train models
status_agent.train(df)
solution_agent.train(df)

# Example ticket
ticket_id = "TK0015"
ticket_row = df[df["Ticket ID"] == ticket_id].iloc[0]

issue_text = f"{ticket_row['Issue Category']} - Priority: {ticket_row['Priority']}"
sentiment = sentiment_agent.analyze_sentiment(issue_text)
recommended_solution = solution_agent.recommend_solution(ticket_row['Issue Category'], ticket_row['Priority'], sentiment)
predicted_status = status_agent.predict_status(ticket_row['Issue Category'], ticket_row['Priority'], sentiment)
summary = summarizer_agent.summarize(issue_text)

# Final Output
print(f"\n--- AI Ticket Resolution Summary for Ticket: {ticket_id} ---")
print(f"Issue: {issue_text}")
print(f"Predicted Sentiment: {sentiment}")
print(f"Recommended Solution: {recommended_solution}")
print(f"Predicted Resolution Status: {predicted_status}")
print(f"Summary: {summary}")
