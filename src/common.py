from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

SEED = 42

CLASS_NAMES = [
    "Extremely Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Extremely Positive",
]

LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

CONFIGURATIONS = ["Baseline", "TextBlob", "SentiStrength", "VADER"]

METRIC_COLUMNS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Balanced Accuracy",
    "MCC",
]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NONLETTER_RE = re.compile(r"[^a-zA-Z#\s]")
SPACE_RE = re.compile(r"\s+")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def light_clean(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def full_clean(text: str) -> str:
    text = light_clean(text)
    text = NONLETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    required = {"OriginalTweet", "Sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["Sentiment"]).copy()
    df["label"] = df["Sentiment"].map(LABEL_MAP)
    if df["label"].isna().any():
        unknown = sorted(df.loc[df["label"].isna(), "Sentiment"].unique())
        raise ValueError(f"Unknown sentiment labels: {unknown}")
    df["label"] = df["label"].astype(int)
    return df


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "Accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "Macro Precision": 100.0 * macro_p,
        "Macro Recall": 100.0 * macro_r,
        "Macro F1": 100.0 * macro_f1,
        "Weighted F1": 100.0
        * f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Balanced Accuracy": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def save_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(path, matrix.tocsr())


def load_sparse(path: Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()
