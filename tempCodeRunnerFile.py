import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [stemmer.stem(word) for word in words]
    print(words)
    return ' '.join(words)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
vectorizer = CountVectorizer()

df = pd.read_csv("D:/internship_clg_aiml/internship_project_3-1/archive (6)/sentiment_analysis.csv")
df["sentiment"] = df["sentiment"].map({'positive': 2, 'neutral': 1, 'negative': 0})

df["Time of Tweet"].value_counts()
df["sentiment"].value_counts()
df['text'] = df['text'].apply(preprocess_text)

X = df["text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train_vectorized, y_train)
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)  
f1 = f1_score(y_test, predictions, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
