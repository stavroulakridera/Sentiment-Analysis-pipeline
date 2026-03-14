import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# =====================================
# Create results folder
# =====================================

os.makedirs("results/confusion_matrices", exist_ok=True)


# =====================================
# Load dataset
# =====================================

df = pd.read_csv("dataset.csv")

texts = df["clean_tweet"]
labels = df["Sentiment"]


# =====================================
# TF-IDF
# =====================================

vectorizer = TfidfVectorizer(max_features=1000)

X = vectorizer.fit_transform(texts)

y = labels


# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================
# Machine Learning Models
# =====================================

models = {

    "Logistic_Regression": LogisticRegression(max_iter=200),

    "Linear_SVM": LinearSVC(),

    "Naive_Bayes": MultinomialNB(),

    "Random_Forest": RandomForestClassifier(),

    "Extra_Trees": ExtraTreesClassifier(),

    "Gradient_Boosting": GradientBoostingClassifier(),

    "Decision_Tree": DecisionTreeClassifier(),

    "KNN": KNeighborsClassifier(),

    "SGD": SGDClassifier(),

    "AdaBoost": AdaBoostClassifier()

}


results = []


# =====================================
# Train and Evaluate
# =====================================

for name, model in models.items():

    print("Training:", name)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results.append((name, acc))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{name} Confusion Matrix")

    plt.ylabel("Actual")

    plt.xlabel("Predicted")

    plt.tight_layout()

    plt.savefig(f"results/confusion_matrices/{name}_cm.png")

    plt.close()


# =====================================
# Save Results Table
# =====================================

results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy"])

results_df = results_df.sort_values(by="Accuracy", ascending=False)

results_df.to_csv("results/ml_results.csv", index=False)

print("\nResults saved successfully.")
print(results_df)
