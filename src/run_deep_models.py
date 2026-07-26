from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from common import (
    CLASS_NAMES,
    CONFIGURATIONS,
    SEED,
    evaluate_predictions,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs" / "features"
OUTPUT = ROOT / "outputs" / "metrics"
PREDICTIONS = ROOT / "outputs" / "predictions"

VOCAB_SIZE = 10_000
MAX_LENGTH = 50
EMBEDDING_DIM = 128
DROPOUT = 0.30
EPOCHS = 5
LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 512
BILSTM_BATCH_SIZE = 1024


def build_vocabulary(texts: List[str]) -> Dict[str, int]:
    counts = Counter()
    for text in texts:
        counts.update(text.split())
    most_common = counts.most_common(VOCAB_SIZE - 2)
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    vocabulary.update(
        {token: index + 2 for index, (token, _) in enumerate(most_common)}
    )
    return vocabulary


def encode_texts(
    texts: List[str],
    vocabulary: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = np.zeros((len(texts), MAX_LENGTH), dtype=np.int64)
    lengths = np.ones(len(texts), dtype=np.int64)

    for row, text in enumerate(texts):
        ids = [vocabulary.get(token, 1) for token in text.split()]
        ids = ids[:MAX_LENGTH]
        lengths[row] = max(1, len(ids))
        sequences[row, : len(ids)] = ids

    return sequences, lengths


class DenseNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    def __init__(self, auxiliary_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            padding_idx=0,
        )
        self.conv = nn.Conv1d(
            EMBEDDING_DIM,
            128,
            kernel_size=5,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(128 + auxiliary_dim, 64)
        self.fc2 = nn.Linear(64, len(CLASS_NAMES))

    def forward(self, tokens, lengths, auxiliary):
        embedded = self.embedding(tokens).transpose(1, 2)
        features = torch.relu(self.conv(embedded))
        pooled = torch.max(features, dim=2).values
        if auxiliary.shape[1] > 0:
            pooled = torch.cat([pooled, auxiliary], dim=1)
        hidden = self.dropout(torch.relu(self.fc1(pooled)))
        return self.fc2(hidden)


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        cell_type: str,
        hidden_size: int,
        bidirectional: bool,
        auxiliary_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM,
            padding_idx=0,
        )
        recurrent_cls = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
        }[cell_type]
        self.recurrent = recurrent_cls(
            input_size=EMBEDDING_DIM,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        representation_dim = hidden_size * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(representation_dim + auxiliary_dim, 64)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(64, len(CLASS_NAMES))

    def forward(self, tokens, lengths, auxiliary):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        output = self.recurrent(packed)
        hidden = output[1]
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.recurrent.bidirectional:
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            representation = hidden[-1]

        if auxiliary.shape[1] > 0:
            representation = torch.cat([representation, auxiliary], dim=1)

        hidden_dense = self.dropout(torch.relu(self.fc1(representation)))
        return self.fc2(hidden_dense)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> np.ndarray:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = batch[:-1]
            labels = batch[-1]
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            total_count += len(labels)
        print(f"epoch={epoch + 1}/{EPOCHS}, loss={total_loss / total_count:.4f}")

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[:-1]
            logits = model(*inputs)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
    return np.asarray(predictions)


def feature_paths(configuration: str) -> Tuple[Path, Path]:
    stem = configuration.lower()
    return (
        FEATURES / f"{stem}_train.npz",
        FEATURES / f"{stem}_test.npz",
    )


def auxiliary_from_matrix(configuration: str, split: str) -> np.ndarray:
    if configuration == "Baseline":
        count = np.load(FEATURES / f"y_{split}.npy").shape[0]
        return np.empty((count, 0), dtype=np.float32)

    matrix = load_npz(FEATURES / f"{configuration.lower()}_{split}.npz")
    auxiliary_dim = 2 if configuration == "SentiStrength" else 1
    return matrix[:, -auxiliary_dim:].toarray().astype(np.float32)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    set_seed()

    processed = pd.read_csv(FEATURES / "processed_dataset.csv")
    train_idx = np.load(FEATURES / "train_indices.npy")
    test_idx = np.load(FEATURES / "test_indices.npy")
    y_train = np.load(FEATURES / "y_train.npy")
    y_test = np.load(FEATURES / "y_test.npy")

    train_texts = processed.loc[train_idx, "cleaned_text"].fillna("").tolist()
    test_texts = processed.loc[test_idx, "cleaned_text"].fillna("").tolist()

    vocabulary = build_vocabulary(train_texts)
    train_sequences, train_lengths = encode_texts(train_texts, vocabulary)
    test_sequences, test_lengths = encode_texts(test_texts, vocabulary)

    architectures = [
        ("BiGRU", "GRU", 32, True, DEFAULT_BATCH_SIZE),
        ("BiLSTM", "LSTM", 32, True, BILSTM_BATCH_SIZE),
        ("CNN", None, None, False, DEFAULT_BATCH_SIZE),
        ("Dense Neural Network", None, None, False, DEFAULT_BATCH_SIZE),
        ("GRU", "GRU", 64, False, DEFAULT_BATCH_SIZE),
        ("LSTM", "LSTM", 64, False, DEFAULT_BATCH_SIZE),
        ("Simple RNN", "RNN", 64, False, DEFAULT_BATCH_SIZE),
    ]

    rows = []
    prediction_frame = pd.DataFrame({"y_true": y_test})

    for configuration in CONFIGURATIONS:
        auxiliary_train = auxiliary_from_matrix(configuration, "train")
        auxiliary_test = auxiliary_from_matrix(configuration, "test")

        for algorithm, cell_type, hidden_size, bidirectional, batch_size in architectures:
            set_seed()
            start = time.time()

            if algorithm == "Dense Neural Network":
                train_path, test_path = feature_paths(configuration)
                X_train = load_npz(train_path).toarray().astype(np.float32)
                X_test = load_npz(test_path).toarray().astype(np.float32)

                generator = torch.Generator().manual_seed(SEED)
                train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(X_train),
                        torch.from_numpy(y_train).long(),
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                    generator=generator,
                )
                test_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_test)),
                    batch_size=batch_size,
                    shuffle=False,
                )

                model = DenseNeuralNetwork(X_train.shape[1])

            else:
                generator = torch.Generator().manual_seed(SEED)
                train_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(train_sequences).long(),
                        torch.from_numpy(train_lengths).long(),
                        torch.from_numpy(auxiliary_train).float(),
                        torch.from_numpy(y_train).long(),
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                    generator=generator,
                )
                test_loader = DataLoader(
                    TensorDataset(
                        torch.from_numpy(test_sequences).long(),
                        torch.from_numpy(test_lengths).long(),
                        torch.from_numpy(auxiliary_test).float(),
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                )

                if algorithm == "CNN":
                    model = CNNClassifier(auxiliary_train.shape[1])
                else:
                    model = RecurrentClassifier(
                        cell_type=cell_type,
                        hidden_size=hidden_size,
                        bidirectional=bidirectional,
                        auxiliary_dim=auxiliary_train.shape[1],
                    )

            predictions = train_model(model, train_loader, test_loader)
            metrics = evaluate_predictions(y_test, predictions)

            rows.append(
                {
                    "Configuration": configuration,
                    "Model Family": "Deep Learning",
                    "Algorithm": algorithm,
                    **metrics,
                    "Batch Size": batch_size,
                    "Elapsed Seconds": time.time() - start,
                }
            )

            safe_name = (
                algorithm.lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            prediction_frame[
                f"{configuration.lower()}__{safe_name}"
            ] = predictions

            print(
                f"{configuration:14s} | {algorithm:22s} | "
                f"Accuracy={metrics['Accuracy']:.2f}"
            )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "deep_model_metrics.csv", index=False)
    results.to_excel(OUTPUT / "deep_model_metrics.xlsx", index=False)
    prediction_frame.to_csv(
        PREDICTIONS / "deep_model_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
