# Sentiment-Analysis-pipeline
Hybrid Machine Learning and Deep Learning framework for 5-class Twitter sentiment analysis using TF-IDF, TextBlob, VADER, and SentiStrength features.


# Twitter Sentiment Analysis (5-Class Classification)

This repository contains the implementation of the sentiment analysis framework used in the paper:

**"A Hybrid Deep Learning Framework for Next-Generation Sentiment Analysis"**

The project investigates sentiment classification on Twitter data related to the COVID-19 pandemic using a combination of machine learning and deep learning approaches.

---

## Dataset

The experiments are conducted on the **COVID-19 Twitter Sentiment Dataset**, which contains tweets annotated with five sentiment categories:

- Extremely Negative
- Negative
- Neutral
- Positive
- Extremely Positive

Dataset size: **41,157 tweets**

Source:
https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification

---

## Methods

### Machine Learning Models

The following classical machine learning algorithms were evaluated:

- Logistic Regression
- Linear Support Vector Machine (SVM)
- Naive Bayes
- Random Forest
- Extra Trees
- Gradient Boosting
- Decision Tree
- K-Nearest Neighbors
- Stochastic Gradient Descent (SGD)
- AdaBoost

### Deep Learning Models

Deep neural architectures implemented with TensorFlow:

- Dense Neural Network
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM (BiLSTM)

---

## Feature Representation

Text data is transformed using:

- **TF–IDF vectorization**
- **Lexicon-based sentiment scores**
  - TextBlob
  - VADER
  - SentiStrength

These features are used to investigate the impact of sentiment lexicons on classification performance.

---

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Repository Structure
