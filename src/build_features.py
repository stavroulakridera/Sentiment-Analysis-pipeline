from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

from common import (
    SEED,
    full_clean,
    light_clean,
    load_dataset,
    save_sparse,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Corona_NLP_train.csv"
VADER_LEXICON = ROOT / "vader_lexicon.txt"
SENTISTRENGTH_JAR = ROOT / "SentiStrength.jar"
SENTISTRENGTH_DATA = ROOT / "SentiStrengthData"
OUTPUT = ROOT / "outputs" / "features"


def build_clean_vader_lexicon(source: Path, destination: Path) -> Path:
    valid_lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            valid_lines.append(line)
    destination.write_text("\n".join(valid_lines), encoding="utf-8")
    return destination


def textblob_polarity(texts: List[str]) -> np.ndarray:
    return np.asarray(
        [TextBlob(text).sentiment.polarity for text in texts],
        dtype=np.float64,
    ).reshape(-1, 1)


def vader_compound(texts: List[str], lexicon_path: Path) -> np.ndarray:
    analyzer = SentimentIntensityAnalyzer(lexicon_file=str(lexicon_path))
    return np.asarray(
        [analyzer.polarity_scores(text)["compound"] for text in texts],
        dtype=np.float64,
    ).reshape(-1, 1)


def sentistrength_scores(
    texts: List[str],
    jar_path: Path,
    data_path: Path,
) -> np.ndarray:
    if not jar_path.exists():
        raise FileNotFoundError(f"Missing SentiStrength jar: {jar_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing SentiStrength data directory: {data_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "sentistrength_input.txt"
        input_path.write_text(
            "\n".join(text.replace("\n", " ") for text in texts),
            encoding="utf-8",
        )

        command = [
            "java",
            "-jar",
            str(jar_path),
            "sentidata",
            str(data_path) + "/",
            "input",
            str(input_path),
            "outputFolder",
            str(temp_dir) + "/",
            "overwrite",
        ]
        subprocess.run(command, check=True)

        candidates = list(temp_dir.glob("*0_out.txt")) + list(
            temp_dir.glob("*_out.txt")
        )
        if not candidates:
            raise RuntimeError("SentiStrength did not produce an output file.")

        output_path = candidates[0]
        result = pd.read_csv(
            output_path,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["positive", "negative"],
        )
        if len(result) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} SentiStrength rows, got {len(result)}."
            )
        return result[["positive", "negative"]].to_numpy(dtype=np.float64)


def fit_scale(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    return scaler.fit_transform(train_values), scaler.transform(test_values)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATASET)
    y = df["label"].to_numpy(dtype=np.int64)
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    raw_texts = df["OriginalTweet"].fillna("").astype(str).tolist()
    light_texts = [light_clean(text) for text in raw_texts]
    cleaned_texts = [full_clean(text) for text in raw_texts]

    train_texts = [cleaned_texts[i] for i in train_idx]
    test_texts = [cleaned_texts[i] for i in test_idx]

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_train = vectorizer.fit_transform(train_texts)
    tfidf_test = vectorizer.transform(test_texts)

    word_count = np.asarray(
        [len(text.split()) for text in cleaned_texts],
        dtype=np.float64,
    ).reshape(-1, 1)
    wc_train, wc_test = fit_scale(word_count[train_idx], word_count[test_idx])

    baseline_train = sparse.hstack(
        [tfidf_train, sparse.csr_matrix(wc_train)],
        format="csr",
    )
    baseline_test = sparse.hstack(
        [tfidf_test, sparse.csr_matrix(wc_test)],
        format="csr",
    )

    textblob_all = textblob_polarity(light_texts)
    textblob_train, textblob_test = fit_scale(
        textblob_all[train_idx],
        textblob_all[test_idx],
    )

    clean_lexicon = build_clean_vader_lexicon(
        VADER_LEXICON,
        OUTPUT / "vader_lexicon_clean.txt",
    )
    vader_all = vader_compound(light_texts, clean_lexicon)
    vader_train, vader_test = fit_scale(
        vader_all[train_idx],
        vader_all[test_idx],
    )

    sentistrength_all = sentistrength_scores(
        light_texts,
        SENTISTRENGTH_JAR,
        SENTISTRENGTH_DATA,
    )
    sentistrength_train, sentistrength_test = fit_scale(
        sentistrength_all[train_idx],
        sentistrength_all[test_idx],
    )

    matrices = {
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "textblob_train": sparse.hstack(
            [baseline_train, sparse.csr_matrix(textblob_train)],
            format="csr",
        ),
        "textblob_test": sparse.hstack(
            [baseline_test, sparse.csr_matrix(textblob_test)],
            format="csr",
        ),
        "sentistrength_train": sparse.hstack(
            [baseline_train, sparse.csr_matrix(sentistrength_train)],
            format="csr",
        ),
        "sentistrength_test": sparse.hstack(
            [baseline_test, sparse.csr_matrix(sentistrength_test)],
            format="csr",
        ),
        "vader_train": sparse.hstack(
            [baseline_train, sparse.csr_matrix(vader_train)],
            format="csr",
        ),
        "vader_test": sparse.hstack(
            [baseline_test, sparse.csr_matrix(vader_test)],
            format="csr",
        ),
    }

    for name, matrix in matrices.items():
        save_sparse(OUTPUT / f"{name}.npz", matrix)

    np.save(OUTPUT / "y_train.npy", y[train_idx])
    np.save(OUTPUT / "y_test.npy", y[test_idx])
    np.save(OUTPUT / "train_indices.npy", train_idx)
    np.save(OUTPUT / "test_indices.npy", test_idx)

    pd.DataFrame(
        {
            "index": indices,
            "light_text": light_texts,
            "cleaned_text": cleaned_texts,
            "label": y,
        }
    ).to_csv(OUTPUT / "processed_dataset.csv", index=False)

    print(f"Feature matrices written to: {OUTPUT}")


if __name__ == "__main__":
    main()
