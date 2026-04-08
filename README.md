# Language-independent Entity Linking

Language-independent Entity Linking for Key-Value (KV) relation extraction in document understanding. Instead of relying on semantics-heavy transformer models (LayoutXLM, LayoutLMv3), this approach uses geometric features (distance, direction, angle, bounding box coordinates) combined with a LightGBM classifier to link question/key entities to answer/value entities.

## Introduction

State-of-the-art VIE/KIE models such as LayoutXLM and LayoutLMv3 tend to link entities based more on semantics than geometric layout, leading to mistakes in scenarios where geometric relations are straightforward. This project verifies that geometric features alone — distance, direction, angle, and bounding box coordinates — are sufficient for accurate KV entity linking, and that such a model generalises across languages without retraining.

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: `lightgbm`, `sentence-transformers`, `pandas`, `rich`, `opencv-python`, `matplotlib`.

## Dataset

Place the RFUND dataset under the repo root:

```
datasets/rfund/
    en.train.json  en.val.json
    zh.train.json  zh.val.json
    ...            (ja, es, fr, it, de, pt)

datasets/images/
    de.val/        (document images, optional — used for visualization)
    ...
```

Pre-processed feature pickles and model weights are cached automatically on first run under `src/features/<lang>/` and `src/weights/<lang>/`.

## Training

```bash
cd src/

# Train a single language
python train.py --lang en

# Train all languages sequentially
python train.py --lang all
```

Supported languages: `en`, `zh`, `ja`, `es`, `fr`, `it`, `de`, `pt`.

After training, feature importance charts (gain-based) are saved to `src/images/feature_importance_<lang>.png`. When training all languages, an averaged chart is saved to `src/images/feature_importance_avg.png`.

Logs are saved to `src/logs/train_<timestamp>.log`.

## Evaluation

```bash
cd src/

# One-shot: evaluate each language with its own model
python evaluate.py --lang en --task one-shot
python evaluate.py --lang all --task one-shot

# Zero-shot: evaluate all languages using the EN model
python evaluate.py --task zero-shot
```

**Optional arguments:**

| Argument | Default | Description |
|---|---|---|
| `--lang` | `all` | Language or `all` |
| `--task` | `one-shot` | `one-shot` or `zero-shot` |
| `--split` | `val` | Data split: `train` or `val` |
| `--output` | `outputs` | Root directory for saved results |

Each evaluation run produces:
- **`outputs/<lang>/predictions.json`** — original input JSON format with `relations.kv_entity` replaced by predicted links
- **`outputs/<lang>/images/`** — annotated document images (key boxes in blue, value boxes in red, linking lines), requires images in `datasets/images/<lang>.<split>/`

Logs are saved to `src/logs/evaluate_<timestamp>.log`.

## Prediction & Visualization

To run inference and visualize results on any split without computing metrics:

```bash
cd src/

python predict.py --lang en
python predict.py --lang all --task zero-shot
python predict.py --lang de --split val --output my_output_dir
```

Annotated images are saved to `<output>/<lang>/<split>/`.

## Evaluation Metric

All scripts use **PEneo-style entity-level exact-match** (defined in `evaluate.calculate_kv_metric`):
- Per document: compare predicted `(key_text, value_text)` pairs against ground truth as sets
- Aggregate TP / FP / FN across all documents → global micro Precision / Recall / F1

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

F1-score on RFUND multilingual subsets (language-specific one-shot task). `-` means the model does not provide pre-trained weights for that language.

| Method | EN | ZH | JA | ES | FR | IT | DE | PT |
|---|---|---|---|---|---|---|---|---|
| Donut (BASE) | - | 28.21 | 13.82 | - | - | - | - | - |
| LayoutLMv3 (Chinese BASE) | - | 72.14 | - | - | - | - | - | - |
| PEneo-LayoutLMv3 (Chinese BASE) | - | 85.05 | - | - | - | - | - | - |
| LiLT[InfoXLM] (BASE) | - | 66.50 | 43.98 | 63.85 | 62.60 | 60.57 | 55.13 | 52.96 |
| PEneo-LiLT[InfoXLM] (BASE) | - | 80.51 | 54.59 | 71.43 | 77.49 | 73.62 | 70.11 | 71.43 |
| LayoutXLM (BASE) | - | 64.11 | 40.21 | 66.75 | 67.98 | 63.04 | 58.77 | 59.79 |
| PEneo-LayoutXLM (BASE) | - | 80.41 | 52.81 | 74.56 | 78.11 | 75.17 | 74.06 | 70.81 |
| **Our approach** | **95.01** | **97.06** | **92.32** | **91.91** | **94.94** | **89.48** | **92.21** | **90.30** |
