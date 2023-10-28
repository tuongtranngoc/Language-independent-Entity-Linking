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


from utils.boxes import *
from config.cfg import Configuration as cfg

class Evaluation:
    def __init__(self, args) -> None:
        self.task = args.task
        if self.task == 'zero-shot':
            self.lang = 'en'
        else:
            self.lang = args.lang
        self.clf = pickle.load(open(os.path.join(cfg.dataset.model_path, self.lang, 'clf.pkl'), 'rb'))
        self.scaler = pickle.load(open(os.path.join(cfg.dataset.scaler_path, self.lang, 'scaler.pkl'), 'rb'))


    def preprocess_data(self, lang):
        val_feat_pth = os.path.join(cfg.dataset.features_path, lang, 'val.pkl')
        val_org_pth = os.path.join(cfg.dataset.features_path, lang, 'val_df.pkl')

        features_val = pickle.load(open(val_feat_pth, 'rb'))
        df_val = pickle.load(open(val_org_pth, 'rb'))

        X_val, y_val = features_val.values[:, :-1], features_val.values[:, -1]
        X_val = self.scaler.transform(X_val)

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

    def single_eval(self, lang):
        X_val, y_val, df_val = self.preprocess_data(lang)
        pred_prob = self.clf.predict_proba(X_val)[:, 1]
        y_preds = self.post_process(df_val, pred_prob).is_linking.values.astype(int)
        y_val = y_val.astype(int)
        print("============================= EVALUATION =================================")
        print(f"Language Specific: {lang}")
        print(f"Recall: {recall_score(y_val, y_preds)}")
        print(f"Precision: {precision_score(y_val, y_preds)}")
        print(f"F1-score: {f1_score(y_val, y_preds)}")

    def evaluate(self):
        if self.task == 'one-shot':
            self.single_eval(self.lang)
        elif self.task == 'zero-shot':
            lang = ['de', 'es', 'fr', 'it', 'ja', 'pt', 'zh']
            for l in lang:
                self.single_eval(l)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default=cfg.dataset.lang, type=str, help='Language specific for evaluation')
    parser.add_argument('--task', default='zero-shot', type=str, help='Language specific task for evaluation')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    val = Evaluation(args)
    val.evaluate()