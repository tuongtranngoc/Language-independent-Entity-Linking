from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import json
import pickle
import argparse
from datetime import datetime

import pandas as pd

from config.cfg import Configuration as cfg
from utils.boxes import unnormalize_scale_bbox
from utils.console import get_printer
from utils.visualization import Visualization

LANGS = ['en', 'zh', 'ja', 'es', 'fr', 'it', 'de', 'pt']
IMAGES_ROOT = os.path.join('..', 'datasets', 'images')
console = None


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _kv_pairs_from_df(df: pd.DataFrame, link_col: str) -> dict:
    """Build per-document sets of (k_text, v_text) tuples."""
    linked = df[df[link_col] == 1.0]
    result = {}
    for fname, group in linked.groupby('fname'):
        result[fname] = set(zip(group['k_text'], group['v_text']))
    return result


def calculate_kv_metric(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    PEneo Level-2 metric: per-document exact-match set comparison of
    (k_text, v_text) pairs, aggregated as global micro P/R/F1.

    Returns dict with precision, recall, f1, and per-document detail.
    """
    pred_pairs = _kv_pairs_from_df(pred_df, 'is_linking')
    gt_pairs   = _kv_pairs_from_df(gt_df,   'label')

    all_fnames = set(pred_pairs) | set(gt_pairs)
    total_tp = total_fp = total_fn = 0
    detail = {}

    for fname in all_fnames:
        pred = pred_pairs.get(fname, set())
        gt   = gt_pairs.get(fname, set())

        tp = len(pred & gt)
        fp = len(pred - gt)
        fn = len(gt - pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        detail[fname] = {
            'TP': [p for p in pred if p in gt],
            'FP': [p for p in pred if p not in gt],
            'FN': [g for g in gt  if g not in pred],
        }

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'detail': detail}


# ---------------------------------------------------------------------------
# Debug / output helpers
# ---------------------------------------------------------------------------

def _save_predictions_json(df_pred: pd.DataFrame, lang: str, split: str, output_dir: str):
    """
    Reload the original input JSON and replace kv_entity relations with
    predicted links, then save to <output_dir>/<lang>/predictions.json.
    """
    src_path = os.path.join(cfg.dataset.data_path, f'{lang}.{split}.json')
    if not os.path.exists(src_path):
        console.print_info(f"Source JSON not found, skipping: {src_path}")
        return

    with open(src_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build predicted links per fname: {fname -> [{from_id, to_id}, ...]}
    linked = df_pred[df_pred['is_linking'] == 1.0]
    pred_links = {}
    for fname, group in linked.groupby('fname'):
        pred_links[fname] = [
            {'from_id': int(row.k_id), 'to_id': int(row.v_id)}
            for _, row in group.iterrows()
        ]

    for doc in data['documents']:
        fname = doc['img']['fname']
        doc['relations']['kv_entity'] = pred_links.get(fname, [])

    save_dir = os.path.join(output_dir, lang)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    console.print_info(f"Predictions JSON → {out_path}")


def _save_visualizations(df_pred: pd.DataFrame, lang: str, split: str, output_dir: str):
    """
    Denormalize bounding boxes, load images from datasets/images/<lang>.<split>/,
    draw KV pair predictions, and save annotated images.
    """
    img_dir = os.path.join(IMAGES_ROOT, f'{lang}.{split}')
    if not os.path.isdir(img_dir):
        console.print_info(f"Image directory not found, skipping visualization: {img_dir}")
        return

    # Denormalize boxes from (0–1) back to pixel coordinates
    df_vis = df_pred.copy()
    df_vis['k_box'] = df_vis.apply(lambda x: unnormalize_scale_bbox(x.k_box, x.width, x.height), axis=1)
    df_vis['v_box'] = df_vis.apply(lambda x: unnormalize_scale_bbox(x.v_box, x.width, x.height), axis=1)

    save_dir = os.path.join(output_dir, lang, 'images')
    os.makedirs(save_dir, exist_ok=True)

    saved, skipped = 0, 0
    for fname in df_vis.fname.unique():
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        df_doc = df_vis[df_vis.fname == fname]
        image = Visualization.visualize_ser_re(df_doc, image, save_dir)
        cv2.imwrite(os.path.join(save_dir, fname), image)
        saved += 1

    console.print_info(f"Visualizations saved: {saved}, skipped (no image): {skipped} → {save_dir}")


# ---------------------------------------------------------------------------
# Evaluation class
# ---------------------------------------------------------------------------

class PEneoEvaluation:
    def __init__(self, args) -> None:
        self.task       = args.task
        self.lang       = args.lang
        self.split      = args.split
        self.output_dir = args.output

    def preprocess_data(self, lang, scaler):
        feat_pth = os.path.join(cfg.dataset.features_path, lang, f'{self.split}.pkl')
        df_pth   = os.path.join(cfg.dataset.features_path, lang, f'{self.split}_df.pkl')
        features = pickle.load(open(feat_pth, 'rb'))
        df       = pickle.load(open(df_pth,   'rb'))

        X = features.values[:, :-1]
        X = scaler.transform(X)
        return X, df.reset_index(drop=True)

    def post_process(self, df: pd.DataFrame, pred_prob) -> pd.DataFrame:
        """One value links to exactly one key (highest predicted probability wins)."""
        df = df.copy()
        df['pred_prob'] = pred_prob
        df['is_linking'] = 0.0
        for fname in df.fname.unique():
            df_fname = df[df.fname == fname]
            for v_id in df_fname.v_id.unique():
                df_vid = df_fname[df_fname.v_id == v_id]
                idx_max = df_vid.pred_prob.idxmax()
                df.loc[(df.fname == fname) & (df.v_id == v_id) & (df.index == idx_max), 'is_linking'] = 1.0
        return df

    def _load_model(self, lang):
        clf    = pickle.load(open(os.path.join(cfg.dataset.model_path,  lang, 'clf.pkl'),    'rb'))
        scaler = pickle.load(open(os.path.join(cfg.dataset.scaler_path, lang, 'scaler.pkl'), 'rb'))
        return clf, scaler

    def single_eval(self, lang, clf, scaler):
        X, df_val = self.preprocess_data(lang, scaler)
        pred_prob = clf.predict_proba(X)[:, 1]
        df_pred   = self.post_process(df_val, pred_prob)

        metrics = calculate_kv_metric(pred_df=df_pred, gt_df=df_val)
        console.print_info(f"Evaluation — language: {lang}")
        console.print_score_result("Precision", f"{metrics['precision']:.4f}")
        console.print_score_result("Recall",    f"{metrics['recall']:.4f}")
        console.print_score_result("F1-score",  f"{metrics['f1']:.4f}")

        _save_predictions_json(df_pred, lang, self.split, self.output_dir)
        _save_visualizations(df_pred, lang, self.split, self.output_dir)

    def evaluate(self):
        if self.task == 'one-shot':
            langs = LANGS if self.lang == 'all' else [self.lang]
            for lang in langs:
                clf, scaler = self._load_model(lang)
                self.single_eval(lang, clf, scaler)
        elif self.task == 'zero-shot':
            clf, scaler = self._load_model('en')
            for lang in [l for l in LANGS if l != 'en']:
                self.single_eval(lang, clf, scaler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang',   default='all',      type=str, help='Language (en/zh/ja/es/fr/it/de/pt) or "all"')
    parser.add_argument('--task',   default='one-shot', type=str, help='one-shot or zero-shot')
    parser.add_argument('--split',  default='val',      type=str, help='Data split: train or val')
    parser.add_argument('--output', default='outputs',  type=str, help='Root output directory for JSON and images')
    return parser.parse_args()


if __name__ == '__main__':
    args = cli()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('logs', f'evaluate_{timestamp}.log')
    console = get_printer(log_file=log_file)
    console.print_info(f'Logs → {log_file}')

    val = PEneoEvaluation(args)
    val.evaluate()
