from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
# nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

df = pd.read_csv("D:/internship_clg_aiml/internship_project_3-1/archive (6)/sentiment_analysis.csv")
df["sentiment"] = df["sentiment"].map({'positive': 2, 'neutral': 1, 'negative': 0})

compound_scores = [analyzer.polarity_scores(text)['compound'] for text in df['text']]
df['compound_score'] = compound_scores
def vader_sentiment(score):
    if score >= 0.05:
        return 2  # positive
    elif score <= -0.05:
        return 0  # negative
    else:
        return 1  # neutral
df['vader_sentiment'] = df['compound_score'].apply(vader_sentiment)
accuracy_score=accuracy_score(df['sentiment'], df['vader_sentiment'])
f1_score_value=f1_score(df['sentiment'], df['vader_sentiment'], average='weighted')
print(f"VADER Sentiment Analysis - Accuracy: {accuracy_score}") 
print(f"VADER Sentiment Analysis - F1 Score: {f1_score_value}")

