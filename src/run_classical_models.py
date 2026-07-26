from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import CONFIGURATIONS, SEED, evaluate_predictions, load_sparse

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs" / "features"
OUTPUT = ROOT / "outputs" / "metrics"
PREDICTIONS = ROOT / "outputs" / "predictions"


def models() -> Dict[str, object]:
    return {
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(random_state=SEED),
        "Histogram-Based Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_iter=20,
            max_leaf_nodes=15,
            max_bins=63,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=SEED,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Linear SVM": LinearSVC(random_state=SEED),
        "Logistic Regression": LogisticRegression(
            max_iter=200,
            random_state=SEED,
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Passive-Aggressive Classifier": PassiveAggressiveClassifier(
            max_iter=1000,
            tol=1e-3,
            random_state=SEED,
        ),
        "Perceptron": Perceptron(
            max_iter=1000,
            tol=1e-3,
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Ridge Classifier": RidgeClassifier(),
        "SGD Classifier": SGDClassifier(random_state=SEED),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)

    y_train = np.load(FEATURES / "y_train.npy")
    y_test = np.load(FEATURES / "y_test.npy")

    rows = []
    prediction_frame = pd.DataFrame({"y_true": y_test})

    for configuration in CONFIGURATIONS:
        stem = configuration.lower()
        X_train = load_sparse(FEATURES / f"{stem}_train.npz")
        X_test = load_sparse(FEATURES / f"{stem}_test.npz")

        for algorithm, model in models().items():
            start = time.time()

            if algorithm == "Histogram-Based Gradient Boosting":
                train_input = X_train.toarray()
                test_input = X_test.toarray()
            else:
                train_input = X_train
                test_input = X_test

            model.fit(train_input, y_train)
            predictions = model.predict(test_input)

            metrics = evaluate_predictions(y_test, predictions)
            rows.append(
                {
                    "Configuration": configuration,
                    "Model Family": "Classical ML",
                    "Algorithm": algorithm,
                    **metrics,
                    "Elapsed Seconds": time.time() - start,
                }
            )

            safe_name = (
                algorithm.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            prediction_frame[f"{stem}__{safe_name}"] = predictions

            print(
                f"{configuration:14s} | {algorithm:38s} | "
                f"Accuracy={metrics['Accuracy']:.2f}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "classical_model_metrics.csv", index=False)
    results.to_excel(OUTPUT / "classical_model_metrics.xlsx", index=False)
    prediction_frame.to_csv(
        PREDICTIONS / "classical_model_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
