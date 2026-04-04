# Language-independent Entity Linking

Language-independent Entity Linking for Key-Value (KV) relation extraction in document understanding. Instead of relying on semantics-heavy transformer models (LayoutXLM, LayoutLMv3), this approach uses geometric features (distance, direction, angle, bounding box coordinates) combined with a LightGBM classifier to link question/key entities to answer/value entities.

## Introduction

State-of-the-art VIE/KIE models such as LayoutXLM and LayoutLMv3 tend to link entities based more on semantics than geometric layout, leading to mistakes in scenarios where geometric relations are straightforward. This project verifies that geometric features alone — distance, direction, angle, and bounding box coordinates — are sufficient for accurate KV entity linking, and that such a model generalises across languages without retraining.

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `lightgbm`, `sentence-transformers`, `pandas`, `rich`, `opencv-python`.

## Dataset

Place the RFUND dataset under the repo root:

```
datasets/rfund/
    en.train.json  en.val.json
    zh.train.json  zh.val.json
    ...            (ja, es, fr, it, de, pt)
```

Pre-processed feature pickles and model weights are cached automatically on first run under `src/features/<lang>/` and `src/weights/<lang>/`.

## Training

```bash
cd src/

# Train all languages (default)
python train.py

# Train a single language
python train.py --lang en
```

Supported languages: `en`, `zh`, `ja`, `es`, `fr`, `it`, `de`, `pt`.

Logs are saved to `src/logs/train_<timestamp>.log`.

## Evaluation

```bash
cd src/

# Evaluate all languages (default)
python evaluate.py

# Evaluate a single language
python evaluate.py --lang en
```

Logs are saved to `src/logs/evaluate_<timestamp>.log`.

## Architecture

1. **Feature engineering** — 21 geometric + embedding features per KV candidate pair:
   - `fe1`–`fe8`: absolute coordinate differences between box corners
   - `fe9`–`fe12`: box width/height
   - `fe13`–`fe17`: angles between corner pairs and centres (`cal_degrees`)
   - `fe18`: gap distance between boxes (`boxes_distance`)
   - `fe19`: Euclidean distance between centres
   - `fe20`/`fe21`: scalar text embedding from `all-MiniLM-L6-v2` (abs mean)

2. **Classifier** — `LGBMClassifier` (binary), one model per language, saved to `weights/<lang>/clf.pkl`.

3. **Post-processing** — each value entity is assigned to the key with the highest predicted probability (one-to-one constraint per document).

## Experiments

F1-score on RFUND multilingual subsets (language-specific task). `-` means the model does not provide pre-trained weights for that language.

| Method | ZH | JA | ES | FR | IT | DE | PT |
|---|---|---|---|---|---|---|---|
| Donut (BASE) | 28.21 | 13.82 | - | - | - | - | - |
| LayoutLMv3 (Chinese BASE) | 72.14 | - | - | - | - | - | - |
| PEneo-LayoutLMv3 (Chinese BASE) | 85.05 | - | - | - | - | - | - |
| LiLT[InfoXLM] (BASE) | 66.50 | 43.98 | 63.85 | 62.60 | 60.57 | 55.13 | 52.96 |
| PEneo-LiLT[InfoXLM] (BASE) | 80.51 | 54.59 | 71.43 | 77.49 | 73.62 | 70.11 | 71.43 |
| LayoutXLM (BASE) | 64.11 | 40.21 | 66.75 | 67.98 | 63.04 | 58.77 | 59.79 |
| PEneo-LayoutXLM (BASE) | 80.41 | 52.81 | 74.56 | 78.11 | 75.17 | 74.06 | 70.81 |
| **Our approach** | **97.06** | **92.32** | **91.91** | **94.94** | **89.48** | **92.21** | **90.30** |
