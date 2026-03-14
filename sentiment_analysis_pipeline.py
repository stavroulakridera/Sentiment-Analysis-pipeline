# ============================================
# Twitter Sentiment Analysis Pipeline
# Hybrid ML + Deep Learning Framework
# ============================================

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Sentiment tools
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("dataset.csv")

print("Dataset shape:", df.shape)

# ============================================
# Basic Preprocessing
# ============================================

df["clean_tweet"] = df["OriginalTweet"].astype(str).str.lower()

df["word_count"] = df["clean_tweet"].apply(lambda x: len(x.split()))

# ============================================
# TextBlob Sentiment
# ============================================

def textblob_score(text):
    return TextBlob(text).sentiment.polarity

df["textblob_score"] = df["clean_tweet"].apply(textblob_score)

# ============================================
# VADER Sentiment
# ============================================

analyzer = SentimentIntensityAnalyzer()

def vader_score(text):
    return analyzer.polarity_scores(text)["compound"]

df["vader_score"] = df["clean_tweet"].apply(vader_score)

# ============================================
# TF-IDF Feature Extraction
# ============================================

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(df["clean_tweet"])

y = df["Sentiment"]

# ============================================
# Train / Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ============================================
# Machine Learning Models
# ============================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=200),

    "Linear SVM": LinearSVC(),

    "Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(),

    "Extra Trees": ExtraTreesClassifier(),

    "Gradient Boosting": GradientBoostingClassifier(),

    "Decision Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "SGD": SGDClassifier(),

    "AdaBoost": AdaBoostClassifier()
}

results = {}

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

# ============================================
# ML Results Table
# ============================================

results_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Accuracy"])

results_df = results_df.sort_values(by="Accuracy", ascending=False)

print(results_df)

# ============================================
# Confusion Matrices
# ============================================

for name, model in models.items():

    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(name + " Confusion Matrix")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.show()

# ============================================
# Deep Learning Models
# ============================================

texts = df["clean_tweet"].astype(str)

tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

max_len = 50

X_seq = pad_sequences(sequences, maxlen=max_len)

# Encode labels
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(df["Sentiment"])

y_dl = to_categorical(y_encoded)

# Train/Test split

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X_seq, y_dl, test_size=0.2, random_state=42
)

# ============================================
# Dense Neural Network
# ============================================

model_dense = Sequential()

model_dense.add(Embedding(10000, 128, input_length=max_len))

model_dense.add(GlobalAveragePooling1D())

model_dense.add(Dense(64, activation="relu"))

model_dense.add(Dense(5, activation="softmax"))

model_dense.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_dense.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32)

loss, acc = model_dense.evaluate(X_test_dl, y_test_dl)

print("Dense NN accuracy:", acc)

# ============================================
# CNN Model
# ============================================

model_cnn = Sequential()

model_cnn.add(Embedding(10000, 128, input_length=max_len))

model_cnn.add(Conv1D(128, 5, activation="relu"))

model_cnn.add(MaxPooling1D())

model_cnn.add(GlobalAveragePooling1D())

model_cnn.add(Dense(64, activation="relu"))

model_cnn.add(Dense(5, activation="softmax"))

model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_cnn.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32)

loss, acc = model_cnn.evaluate(X_test_dl, y_test_dl)

print("CNN accuracy:", acc)

# ============================================
# LSTM Model
# ============================================

model_lstm = Sequential()

model_lstm.add(Embedding(10000, 128, input_length=max_len))

model_lstm.add(LSTM(128))

model_lstm.add(Dense(5, activation="softmax"))

model_lstm.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_lstm.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32)

loss, acc = model_lstm.evaluate(X_test_dl, y_test_dl)

print("LSTM accuracy:", acc)

# ============================================
# GRU Model
# ============================================

model_gru = Sequential()

model_gru.add(Embedding(10000, 128, input_length=max_len))

model_gru.add(GRU(128))

model_gru.add(Dense(5, activation="softmax"))

model_gru.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_gru.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32)

loss, acc = model_gru.evaluate(X_test_dl, y_test_dl)

print("GRU accuracy:", acc)

# ============================================
# BiLSTM Model
# ============================================

model_bilstm = Sequential()

model_bilstm.add(Embedding(10000, 128, input_length=max_len))

model_bilstm.add(Bidirectional(LSTM(128)))

model_bilstm.add(Dense(5, activation="softmax"))

model_bilstm.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_bilstm.fit(X_train_dl, y_train_dl, epochs=5, batch_size=32)

loss, acc = model_bilstm.evaluate(X_test_dl, y_test_dl)

print("BiLSTM accuracy:", acc)

print("Pipeline completed successfully.")
