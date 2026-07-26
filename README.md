# Lexicon-Enhanced Sentiment Classification

This repository reproduces the experiments reported in the accompanying paper.

## Experimental design

- Dataset: `Corona_NLP_train.csv`
- Task: five-class sentiment classification
- Split: stratified 80/20 train/test split
- Random seed: 42
- Classical models: 15
- Deep learning models: 7
- Feature configurations:
  - Baseline
  - TextBlob
  - SentiStrength
  - VADER
- Total experiments: 88

## Repository structure

```text
src/
  build_features.py
  run_classical_models.py
  run_deep_models.py
  statistical_analysis.py
  confusion_matrices.py
requirements.txt
```

## Required files

Place the following files in the project root:

```text
Corona_NLP_train.csv
vader_lexicon.txt
SentiStrength.jar
SentiStrengthData/
```

The `SentiStrengthData/` directory must contain the English SentiStrength data files.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Java is required for SentiStrength.

## Execution order

```bash
python src/build_features.py
python src/run_classical_models.py
python src/run_deep_models.py
python src/statistical_analysis.py
python src/confusion_matrices.py
```

Generated files are written to:

```text
outputs/
  features/
  metrics/
  statistics/
  figures/
  predictions/
```

## Reproducibility notes

- TF-IDF is fitted only on the training partition.
- Auxiliary scalers are fitted only on the training partition.
- All configurations use the same stratified train/test split.
- Recurrent models use padding-aware processing through
  `pack_padded_sequence`.
- BiLSTM uses 32 units per direction and batch size 1,024.
- The remaining neural models use batch size 512.
- Neural models are trained for five epochs with Adam and learning rate 0.001.
- Holm correction is applied separately to the six Wilcoxon comparisons
  performed for each evaluation metric.
