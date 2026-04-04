from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import cv2
import json
import pickle
import argparse
import itertools
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler

from utils.boxes import *
from utils.console import get_printer
from config.cfg import Configuration as cfg
from modules.kv_embedding import KVEmbedding

from sklearn.metrics import recall_score, precision_score, f1_score

LANGS = ['en', 'zh', 'ja', 'es', 'fr', 'it', 'de', 'pt']
console = None  # initialised in __main__ with a log file


class Trainer(object):
    def __init__(self, args) -> None:
        self.kv_embed = KVEmbedding()
        self.dirname = cfg.dataset.image_path
        self.cols = ['k_id', 'k_text', 'k_box', 'v_id', 'v_text', 'v_box', 'k_embed', 'v_embed', 'width', 'height', 'fname']
        self.lang = args.lang
        
    def load_data(self, type_data='train'):
        df = []
        label_path = os.path.join(cfg.dataset.data_path, f"{self.lang}.{type_data}.json")
        with open(label_path, 'r') as f_json:
            data = json.load(f_json)
        for doc in tqdm(data['documents']):
            f_name = doc['img']['fname']
            h, w = doc['img']['height'], doc['img']['width']

            # Build entity lookup: id -> {text, box, label}
            # Merge multiple lines per entity into a single span
            entities = {}
            for ent in doc['entities']:
                text = ' '.join(line['text'] for line in ent['lines'])
                bboxes = [line['bbox'] for line in ent['lines']]
                box = [
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                ]
                entities[ent['id']] = {'text': text, 'box': box, 'label': ent['label']}

            kv_relations = doc['relations'].get('kv_entity', [])
            if not kv_relations:
                continue

            # Make linking data (positive pairs)
            re = []
            for rel in kv_relations:
                k_id, v_id = rel['from_id'], rel['to_id']
                if k_id not in entities or v_id not in entities:
                    continue
                k, v = entities[k_id], entities[v_id]
                re.append({
                    'k_id': k_id,
                    'k_text': k['text'],
                    'k_box': k['box'],
                    'k_embed': self.kv_embed.embedding(k['text']),
                    'v_id': v_id,
                    'v_text': v['text'],
                    'v_box': v['box'],
                    'v_embed': self.kv_embed.embedding(v['text']),
                    'width': w,
                    'height': h,
                    'fname': f_name,
                })
            re = pd.DataFrame(re)
            if re.shape[0] == 0:
                continue

            non_re = []
            # Make non-linking data
            for (i, k_id, k_box, k_text, k_embed, w, h, fname), (j, v_id, v_box, v_text, v_embed) in itertools.product(
                re[['k_id', 'k_box', 'k_text', 'k_embed', 'width', 'height', 'fname']].to_records(index=True),
                re[['v_id', 'v_box', 'v_text', 'v_embed']].to_records(index=True)):
                if i == j: continue
                non_re.append({
                    'k_box': k_box,
                    'v_box': v_box,
                    'k_id': k_id,
                    'v_id': v_id,
                    'k_text': k_text,
                    'v_text': v_text,
                    'k_embed': k_embed,
                    'v_embed': v_embed,
                    'width': w,
                    'height': h,
                    'fname': fname,
                })
            non_re = pd.DataFrame(non_re).reset_index(drop=True)
            non_re['label'] = 0.0

            re = re[self.cols].copy()
            re['label'] = 1.0
            re_total = pd.concat([re, non_re])
            df.append(re_total)
        return pd.concat(df)
    
    def make_features(self, df:pd.DataFrame):
        console.print_info("Making features ...")
        df.k_box = df.apply(lambda x: normalize_scale_bbox(x.k_box, x.width, x.height), axis=1)
        df.v_box = df.apply(lambda x:normalize_scale_bbox(x.v_box, x.width, x.height), axis=1)
        k_features = pd.DataFrame(df.k_box.tolist(), index=df.index, columns=['k_' + s for s in ['x1', 'y1', 'x2', 'y2']])
        v_features = pd.DataFrame(df.v_box.tolist(), index=df.index, columns=['v_' + s for s in ['x1', 'y1', 'x2', 'y2']])
        
        df = pd.concat([k_features, v_features, df[self.cols], df['label']], axis=1)
        
        df['k_cx'] = df.k_x1.add(df.k_x2).div(2)
        df['k_cy'] = df.k_y1.add(df.k_y2).div(2)
        
        df['v_cx'] = df.v_x1.add(df.v_x2).div(2)
        df['v_cy'] = df.v_y1.add(df.v_y2).div(2)
        
        df['fe1'] = abs(df.v_x1 - df.k_x1)
        df['fe2'] = abs(df.v_y1 - df.k_y1)
        df['fe3'] = abs(df.v_x1 - df.k_x2)
        df['fe4'] = abs(df.v_y1 - df.k_y2)
        df['fe5'] = abs(df.v_x2 - df.k_x1)
        df['fe6'] = abs(df.v_y2 - df.k_y1)
        df['fe7'] = abs(df.v_x2 - df.k_x2)
        df['fe8'] = abs(df.v_y2 - df.k_y2)
        df['fe9'] = abs(df.v_x2 - df.v_x1)
        df['fe10'] = abs(df.v_y2 - df.v_y1)
        df['fe11'] = abs(df.k_x2 - df.k_x1)
        df['fe12'] = abs(df.k_y2 - df.k_y1)
        
        df['fe13'] = df.apply(lambda x: cal_degrees([x.k_x1, x.k_y1], [x.v_x1, x.v_y1]), axis=1)
        df['fe14'] = df.apply(lambda x: cal_degrees([x.k_x2, x.k_y1], [x.v_x2, x.v_y1]), axis=1)
        df['fe15'] = df.apply(lambda x: cal_degrees([x.k_x2, x.k_y2], [x.v_x2, x.v_y2]), axis=1)
        df['fe16'] = df.apply(lambda x: cal_degrees([x.k_x1, x.k_y2], [x.v_x1, x.v_y2]), axis=1)
        df['fe17'] = df.apply(lambda x: cal_degrees([x['k_cx'], x['k_cy']], [x['v_cx'], x['v_cy']]), axis=1)

        df['fe18'] = df.apply(lambda x: boxes_distance([x.k_x1-x.v_x2, x.k_y2-x.v_y1],[x.v_x1-x.k_x2, x.v_y2-x.k_y1]), axis=1)
        df['fe19'] = df.apply(lambda x: dist_points([x.k_cx, x.k_cy], [x.v_cx, x.v_cy]), axis=1)

        df['fe20'] = df['k_embed']
        df['fe21'] = df['v_embed']
        
        cols = [c for c in df.columns if c.startswith('fe')] + ['label']

        return df[cols], df[self.cols]
    
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

    def preprocess_data(self):
        os.makedirs(os.path.join(cfg.dataset.features_path, self.lang), exist_ok=True)
        os.makedirs(os.path.join(cfg.dataset.scaler_path, self.lang), exist_ok=True)
        train_feat_pth = os.path.join(cfg.dataset.features_path, self.lang, 'train.pkl')
        val_feat_pth = os.path.join(cfg.dataset.features_path, self.lang, 'val.pkl')
        train_org_pth = os.path.join(cfg.dataset.features_path, self.lang, 'train_df.pkl')
        val_org_pth = os.path.join(cfg.dataset.features_path, self.lang, 'val_df.pkl')
        scaler_pth = os.path.join(cfg.dataset.scaler_path, self.lang, 'scaler.pkl')
        scaler = StandardScaler()
        
        if os.path.exists(train_feat_pth):
            console.print_info("Loading features training data ...")
            features_train = pickle.load(open(train_feat_pth, 'rb'))
            df_train = pickle.load(open(train_org_pth, 'rb'))
        else:
            console.print_info("Loading training data ...")
            df_train = self.load_data(type_data='train')
            features_train, __ = self.make_features(df_train)
            pickle.dump(df_train, open(train_org_pth, 'wb'))
            pickle.dump(features_train, open(train_feat_pth, 'wb'))

        if os.path.exists(val_feat_pth):
            console.print_info("Loading features valid data ...")
            features_val = pickle.load(open(val_feat_pth, 'rb'))
            df_val = pickle.load(open(val_org_pth, 'rb'))
        else:
            console.print_info("Loading valid data ...")
            df_val = self.load_data(type_data='val')
            features_val, __ = self.make_features(df_val)
            pickle.dump(df_val, open(val_org_pth, 'wb'))
            pickle.dump(features_val, open(val_feat_pth, 'wb'))
            
        X_train, y_train = features_train.values[:, :-1], features_train.values[:, -1].astype(int)
        if os.path.exists(scaler_pth):
            scaler = pickle.load(open(scaler_pth, 'rb'))
            X_train = scaler.transform(X_train)
        else:
            X_train = scaler.fit_transform(X_train)
            pickle.dump(scaler, open(scaler_pth, 'wb'))
        
        X_val, y_val = features_val.values[:, :-1], features_val.values[:, -1].astype(int)
        X_val = scaler.transform(X_val)
        
        return (X_train, y_train), (X_val, y_val), (df_train.reset_index(drop=True), df_val.reset_index(drop=True))

        
    def train(self):
        os.makedirs(os.path.join(cfg.dataset.model_path, self.lang), exist_ok=True)
        train_data, val_data, data_df = self.preprocess_data()
        X_train, y_train = train_data
        X_val, y_val = val_data
        train_df, val_df = data_df
        
        console.print_info(f"X_train: {X_train.shape}  |  X_val: {X_val.shape}")

        if not os.path.exists(os.path.join(cfg.dataset.model_path, self.lang, 'clf.pkl')):
            console.print_info("Training model ...")
            if os.path.exists(cfg.dataset.params):
                console.print_info("Loading tuned params ...")
                params = json.load(open(cfg.dataset.params, 'r', encoding='utf-8'))
            else:
                console.print_info("Using default params ...")
                params = {
                    'random_state': 1997,
                    'n_estimators': 200,
                    'n_jobs': 15,
                    'max_depth': 10,
                }
            # clf = xgb.XGBClassifier(objective="binary:logistic", **params)
            clf = lgbm.LGBMClassifier(objective='binary', **params)
            clf.fit(X_train, y_train)
            console.print_info("Saving model ...")
            with open(os.path.join(cfg.dataset.model_path, self.lang, 'clf.pkl'), 'wb') as f_cls:
                pickle.dump(clf, f_cls, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            clf = pickle.load(open(os.path.join(cfg.dataset.model_path, self.lang, 'clf.pkl'), 'rb'))

        pred_prob = clf.predict_proba(X_val)[:, 1]
        y_preds = self.post_process(val_df, pred_prob).is_linking.values.astype(int)
        y_val = y_val.astype(int)
        console.print_info(f"Evaluation — language: {self.lang}")
        console.print_score_result("Recall",    f"{recall_score(y_val, y_preds):.4f}")
        console.print_score_result("Precision", f"{precision_score(y_val, y_preds):.4f}")
        console.print_score_result("F1-score",  f"{f1_score(y_val, y_preds):.4f}")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='all', type=str,
                        help='Language to train (en/zh/ja/es/fr/it/de/pt), or "all" for every language')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"train_{timestamp}.log")
    console = get_printer(log_file=log_file)
    console.print_info(f"Logs → {log_file}")

    trainer = Trainer(args)
    langs = LANGS if args.lang == 'all' else [args.lang]
    for lang in langs:
        console.print_info(f"Training [{lang}]")
        trainer.lang = lang
        trainer.train()
