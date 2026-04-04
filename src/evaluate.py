from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm
import pandas as pd
import itertools
import argparse
import pickle
import json
import os


import os
from datetime import datetime

from utils.boxes import *
from utils.console import get_printer
from config.cfg import Configuration as cfg

LANGS = ['en', 'zh', 'ja', 'es', 'fr', 'it', 'de', 'pt']
console = None  # initialised in __main__ with a log file


class Evaluation:
    def __init__(self, args) -> None:
        self.task = args.task
        self.lang = args.lang

    def preprocess_data(self, lang, scaler):
        val_feat_pth = os.path.join(cfg.dataset.features_path, lang, 'val.pkl')
        val_org_pth = os.path.join(cfg.dataset.features_path, lang, 'val_df.pkl')

        features_val = pickle.load(open(val_feat_pth, 'rb'))
        df_val = pickle.load(open(val_org_pth, 'rb'))

        X_val, y_val = features_val.values[:, :-1], features_val.values[:, -1]
        X_val = scaler.transform(X_val)

        return X_val, y_val, df_val

    def post_process(self, df: pd.DataFrame, pred_prob):
        # one value only links to one key but one key can link to many value
        df['pred_prob'] = pred_prob
        df['is_linking'] = 0.0

        fnames = df.fname.unique().tolist()
        for fname in fnames:
            df_fname = df[df.fname==fname]
            v_ids = df_fname.v_id.unique().tolist()
            for v_id in v_ids:
                df_vid = df_fname[df_fname.v_id==v_id]
                idx_max = df_vid.pred_prob.idxmax()
                df.loc[(df.fname==fname)&(df.v_id==v_id)&(df.index==idx_max), 'is_linking'] = 1.0
        return df

    def single_eval(self, lang, clf, scaler):
        X_val, y_val, df_val = self.preprocess_data(lang, scaler)
        pred_prob = clf.predict_proba(X_val)[:, 1]
        y_preds = self.post_process(df_val, pred_prob).is_linking.values.astype(int)
        y_val = y_val.astype(int)
        console.print_info(f"Evaluation — language: {lang}")
        console.print_score_result("Recall",    f"{recall_score(y_val, y_preds):.4f}")
        console.print_score_result("Precision", f"{precision_score(y_val, y_preds):.4f}")
        console.print_score_result("F1-score",  f"{f1_score(y_val, y_preds):.4f}")

    def _load_model(self, lang):
        clf = pickle.load(open(os.path.join(cfg.dataset.model_path, lang, 'clf.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(cfg.dataset.scaler_path, lang, 'scaler.pkl'), 'rb'))
        return clf, scaler

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


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='all', type=str,
                        help='Language to evaluate (en/zh/ja/es/fr/it/de/pt), or "all" for every language')
    parser.add_argument('--task', default='one-shot', type=str, help='Evaluation task: one-shot or zero-shot')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"evaluate_{timestamp}.log")
    console = get_printer(log_file=log_file)
    console.print_info(f"Logs → {log_file}")

    val = Evaluation(args)
    val.evaluate()