from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from common import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "outputs" / "predictions"
OUTPUT = ROOT / "outputs" / "figures"
STATISTICS = ROOT / "outputs" / "statistics"


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str,
) -> None:
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        normalize="true",
    ) * 100.0

    figure, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(matrix, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("True Class")
    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))
    axis.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticklabels(CLASS_NAMES)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1f}",
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis, label="Percentage (%)")
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(figure)


def class_metrics(
    model: str,
    configuration: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for sentiment_class in CLASS_NAMES:
        rows.append(
            {
                "Model": model,
                "Configuration": configuration,
                "Sentiment Class": sentiment_class,
                "Precision": 100.0 * report[sentiment_class]["precision"],
                "Recall": 100.0 * report[sentiment_class]["recall"],
                "F1-Score": 100.0 * report[sentiment_class]["f1-score"],
                "Support": int(report[sentiment_class]["support"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATISTICS.mkdir(parents=True, exist_ok=True)

    classical = pd.read_csv(
        PREDICTIONS / "classical_model_predictions.csv"
    )
    deep = pd.read_csv(
        PREDICTIONS / "deep_model_predictions.csv"
    )

    y_true = classical["y_true"].to_numpy()
    if not np.array_equal(y_true, deep["y_true"].to_numpy()):
        raise ValueError("Classical and deep prediction files use different targets.")

    selected = [
        (
            "AdaBoost",
            "VADER",
            classical["vader__adaboost"].to_numpy(),
            "AdaBoost with VADER-Enhanced Features",
            "confusion_matrix_adaboost_vader",
        ),
        (
            "Dense Neural Network",
            "VADER",
            deep["vader__dense_neural_network"].to_numpy(),
            "Dense Neural Network with VADER-Enhanced Features",
            "confusion_matrix_dense_vader",
        ),
    ]

    per_class_frames = []
    for model, configuration, predictions, title, filename in selected:
        save_confusion_matrix(
            y_true,
            predictions,
            title,
            filename,
        )
        per_class_frames.append(
            class_metrics(
                model,
                configuration,
                y_true,
                predictions,
            )
        )

    per_class = pd.concat(per_class_frames, ignore_index=True)
    per_class.to_csv(
        STATISTICS / "per_class_metrics_best_models.csv",
        index=False,
    )
    per_class.to_excel(
        STATISTICS / "per_class_metrics_best_models.xlsx",
        index=False,
    )

    print(f"Figures written to: {OUTPUT}")


if __name__ == "__main__":
    main()
