# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Language-independent Entity Linking for Key-Value (KV) relation extraction in document understanding. Instead of relying on semantics-heavy transformer models (LayoutXLM, LayoutLMv3), this approach uses geometric features (distance, direction, angle, bounding box coordinates) combined with a LightGBM classifier to link question/key entities to answer/value entities.

## Setup

```bash
pip install -r requirements.txt
```

**Required data layout** (relative to repo root):
- `datasets/rfund/` — raw RFUND JSON files named `<lang>.train.json`, `<lang>.val.json`
- `src/features/<lang>/` — pre-processed feature pickles (auto-generated on first run of `train.py`)
- `src/weights/<lang>/` — scaler and model checkpoints (auto-generated on first run of `train.py`)

## Common Commands

All scripts must be run from `src/`:

```bash
cd src/

# Train one language or all
python train.py --lang en        # or zh, ja, es, fr, it, de, pt
python train.py --lang all       # trains all 8 languages; saves per-language + avg feature importance charts

# Evaluate (requires pre-trained model for that language)
python evaluate.py --lang en --task one-shot
python evaluate.py --lang all --task one-shot

# Zero-shot evaluation (uses EN model, evaluates on all other languages)
python evaluate.py --task zero-shot

# Hyperparameter tuning (XGBoost-based Bayesian optimization, separate from LightGBM classifier)
python modules/hyperparams_tuning.py
```

Logs: `src/logs/train_<timestamp>.log`, `src/logs/peneo_evaluate_<timestamp>.log`  
Feature importance charts: `src/images/feature_importance_<lang>.png`, `src/images/feature_importance_avg.png`

## Architecture

### Pipeline

1. **Data loading** (`Trainer.load_data`) — reads RFUND JSON (`doc['entities']` with `lines`, `doc['relations']['kv_entity']`), builds positive KV pairs from annotations and exhaustive negative pairs via `itertools.product`.
2. **Feature engineering** (`Trainer.make_features`) — normalizes bounding boxes, computes 21 features (`fe1`–`fe21`):
   - `fe1`–`fe8`: absolute coordinate differences between box corners
   - `fe9`–`fe12`: box width/height
   - `fe13`–`fe17`: angles (`cal_degrees`) between box corner pairs and centers
   - `fe18`: box gap distance (`boxes_distance`)
   - `fe19`: Euclidean distance between centers
   - `fe20`/`fe21`: scalar text embeddings from `KVEmbedding` (abs mean of `all-MiniLM-L6-v2` sentence embedding)
3. **Scaler** — `StandardScaler` fitted on train set, saved to `weights/<lang>/scaler.pkl`
4. **Classifier** — `LGBMClassifier` (binary), saved to `weights/<lang>/clf.pkl`; optional tuned params loaded from `params_best/params.json`
5. **Post-processing** (`post_process`) — enforces the constraint that each value entity links to exactly one key (highest predicted probability wins per `v_id` per document)

### Evaluation Metric

The metric is **PEneo-style entity-level exact-match** defined in `evaluate.calculate_kv_metric`:
- Per document: compare sets of predicted `(k_text, v_text)` tuples against ground truth
- Aggregate TP/FP/FN across all documents → global micro Precision / Recall / F1
- `train.py` imports `calculate_kv_metric` from `evaluate` — do not add separate metric logic elsewhere

### Key Files

| File | Role |
|------|------|
| [src/config/cfg.py](src/config/cfg.py) | Central config (`Configuration.dataset`); edit paths and default language here |
| [src/evaluate.py](src/evaluate.py) | Owns `calculate_kv_metric` (the single metric); `PEneoEvaluation` class for standalone eval |
| [src/train.py](src/train.py) | `Trainer` class — full train + eval loop; imports metric from `evaluate`; saves feature importance charts |
| [src/modules/kv_embedding.py](src/modules/kv_embedding.py) | Wraps `sentence-transformers` `all-MiniLM-L6-v2`; returns scalar (abs mean of embedding vector) |
| [src/utils/boxes.py](src/utils/boxes.py) | Geometric utilities: angle, distance, normalization |
| [src/modules/hyperparams_tuning.py](src/modules/hyperparams_tuning.py) | Bayesian optimization over XGBoost params (not LightGBM) |

### Configuration

Edit [src/config/cfg.py](src/config/cfg.py) to change paths before training. Key fields:
- `data_path` — directory containing raw JSON files (default: `../datasets/rfund`)
- `features_path` — where pre-processed pickles are cached (default: `features`)
- `model_path` / `scaler_path` — where trained artifacts are saved (default: `weights`)
- `params` — path to tuned hyperparams JSON (`params_best/params.json`)

Features and model checkpoints are cached on first run; delete the relevant pickle files to force recomputation. `evaluate.py` requires features to already exist — run `train.py` first.
