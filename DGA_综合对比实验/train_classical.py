import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import extract_features, FEATURE_NAMES, build_bigram_freq


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {'domain','label'} <= set(df.columns)
    return df


def build_bigram_model(legit_domains: List[str]):
    unique_legit = tuple(sorted(set(legit_domains)))
    return build_bigram_freq(unique_legit)


def vectorize(domains: List[str], bigram_probs: Dict[str, float]) -> np.ndarray:
    return np.array([extract_features(d, bigram_probs) for d in domains], dtype=float)


def train_models(csv: str, out: str, test_size: float, rf_est: int, xgb_est: int, random_state: int = 42):
    df = load_data(csv)
    bigram_probs = build_bigram_model(df[df.label=='legit']['domain'].tolist())
    X = vectorize(df['domain'].tolist(), bigram_probs)
    y = (df['label']=='dga').astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rf = RandomForestClassifier(n_estimators=rf_est, n_jobs=-1, random_state=random_state, class_weight='balanced')
    rf.fit(X_train, y_train)
    xgb = XGBClassifier(
        n_estimators=xgb_est, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8,
        objective='binary:logistic', eval_metric='logloss', n_jobs=-1, random_state=random_state, verbosity=0
    )
    xgb.fit(X_train, y_train)

    def eval_model(model, name):
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model,'predict_proba') else y_pred.astype(float)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        print(f"=== {name} Metrics ===")
        for k,v in metrics.items():
            print(f"{k}: {v:.4f}")
        return metrics

    rf_metrics = eval_model(rf, 'RandomForest')
    xgb_metrics = eval_model(xgb, 'XGBoost')

    os.makedirs(out, exist_ok=True)
    with open(Path(out)/'bigram_probs.json','w',encoding='utf-8') as f:
        json.dump(bigram_probs, f)
    joblib.dump({'model': rf, 'feature_names': FEATURE_NAMES}, Path(out)/'rf_model.joblib')
    joblib.dump({'model': xgb, 'feature_names': FEATURE_NAMES}, Path(out)/'xgb_model.joblib')
    print(f"Saved artifacts to {out}")
    return rf_metrics, xgb_metrics


def parse_args():
    ap = argparse.ArgumentParser(description='Train RandomForest and XGBoost on DGA dataset')
    # default now points to local copy to remove external folder dependency
    ap.add_argument('--csv', default='dga_training_data.csv')
    ap.add_argument('--out', default='artifacts_classical')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--rf-est', type=int, default=150)
    ap.add_argument('--xgb-est', type=int, default=300)
    return ap.parse_args()


def main():
    args = parse_args()
    train_models(args.csv, args.out, args.test_size, args.rf_est, args.xgb_est)

if __name__ == '__main__':
    main()
