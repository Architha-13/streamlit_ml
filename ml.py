import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("D:/internship_clg_aiml/internship_project_3-1/archive (6)/sentiment_analysis.csv")
df["sentiment"] = df["sentiment"].map({'positive': 2, 'neutral': 1, 'negative': 0})


df["Time of Tweet"].value_counts()
df["sentiment"].value_counts()

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    words = [stemmer.stem(word) for word in words]
    print(words)
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X = df["text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train_vectorized, y_train)
print(X_test_vectorized)
# print(X_train_vectorized)

from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42,  multi_class='multinomial',class_weight='balanced')
model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')



model.fit(X_train, y_train)

predictions = model.predict(X_test_vectorized)

for text, label in zip(X_test, predictions):
    print(f"'{text}' → {label}")

import streamlit as st
st.title("Sentiment Analysis App 😊")
st.write("Train a model and predict sentiment from text input.")

# st.write(f"Model Accuracy on Test Set: {acc*100:.2f}%")

    # User input for prediction
user_input = st.text_area("Enter text to predict sentiment:")

if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)[0]
            emoji_dict = {2: "Positive 😃", 1: "Neutral 😐", 0: "Negative 😡"}

            # Using st.success
            st.success(f"Predicted Sentiment: {emoji_dict.get(prediction, '')}")

            # Or using st.markdown (better for emoji rendering)
            # st.markdown(f"**Predicted Sentiment:** {prediction} {emoji_dict.get(prediction, '')}")
