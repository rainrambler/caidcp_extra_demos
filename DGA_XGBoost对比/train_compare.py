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
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# Reuse original feature extraction by relative import if running at repo root
import sys
# Ensure parent dir is in path to import features from original if needed
PARENT = Path(__file__).resolve().parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

try:
    from 随机森林检测DGA.features import extract_features, FEATURE_NAMES, build_bigram_freq  # type: ignore
except ModuleNotFoundError:
    # fallback: local features wrapper (thin) if provided
    from features import extract_features, FEATURE_NAMES, build_bigram_freq  # type: ignore


MODEL_DIR = 'model_artifacts'


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert {'domain','label'} <= set(df.columns)
    return df


def build_bigram_model(legit_domains: List[str]):
    unique_legit = tuple(sorted(set(legit_domains)))
    return build_bigram_freq(unique_legit)


def vectorize(domains: List[str], bigram_probs: Dict[str, float]) -> np.ndarray:
    return np.array([extract_features(d, bigram_probs) for d in domains], dtype=float)


def evaluate_cv(X, y, build_model_fn, k=5, random_state=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    metrics = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model = build_model_fn()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_te)[:,1]
        else:
            # fallback using decision_function if available
            if hasattr(model, 'decision_function'):
                raw = model.decision_function(X_te)
                # scale to 0-1 via logistic
                y_prob = 1/(1+np.exp(-raw))
            else:
                y_prob = y_pred.astype(float)
        metrics.append({
            'accuracy': accuracy_score(y_te, y_pred),
            'precision': precision_score(y_te, y_pred),
            'recall': recall_score(y_te, y_pred),
            'f1': f1_score(y_te, y_pred),
            'roc_auc': roc_auc_score(y_te, y_prob)
        })
    # aggregate
    agg = {m: np.mean([r[m] for r in metrics]) for m in metrics[0].keys()}
    return agg, metrics


def train_and_save_final(X, y, build_model_fn, name: str, out_dir: str):
    model = build_model_fn()
    model.fit(X, y)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({'model': model, 'feature_names': FEATURE_NAMES}, Path(out_dir)/f'{name}_model.joblib')
    return model


def comparison(csv_path: str, out_dir: str, n_estimators_rf: int, n_estimators_xgb: int, kfold: int):
    df = load_data(csv_path)
    bigram_probs = build_bigram_model(df[df.label == 'legit']['domain'].tolist())
    X = vectorize(df['domain'].tolist(), bigram_probs)
    y = (df['label'] == 'dga').astype(int).values

    print(f"Dataset shape: {X.shape}, positive rate={y.mean():.3f}")

    def build_rf():
        return RandomForestClassifier(
            n_estimators=n_estimators_rf, random_state=42, n_jobs=-1, class_weight='balanced'
        )

    def build_xgb():
        return XGBClassifier(
            n_estimators=n_estimators_xgb,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )

    print("Performing Stratified K-Fold cross validation (k={})...".format(kfold))
    rf_cv, rf_details = evaluate_cv(X, y, build_rf, k=kfold)
    xgb_cv, xgb_details = evaluate_cv(X, y, build_xgb, k=kfold)

    def fmt(d):
        return ' | '.join(f"{k}:{v:.4f}" for k,v in d.items())

    print("\n=== CV Average Metrics ===")
    print("RandomForest  ->", fmt(rf_cv))
    print("XGBoost       ->", fmt(xgb_cv))

    # Train final models on full data
    print("\nTraining final models on full dataset...")
    train_and_save_final(X, y, build_rf, 'rf', out_dir)
    train_and_save_final(X, y, build_xgb, 'xgb', out_dir)

    # Save bigram probs for feature extraction reuse
    with open(Path(out_dir)/'bigram_probs.json', 'w', encoding='utf-8') as f:
        json.dump(bigram_probs, f)
    print(f"Artifacts saved to {out_dir}")


def predict(domain: str, model_type: str, model_dir: str):
    # Load bigram
    with open(Path(model_dir)/'bigram_probs.json', 'r', encoding='utf-8') as f:
        bigram_probs = json.load(f)
    # Load model
    bundle = joblib.load(Path(model_dir)/f'{model_type}_model.joblib')
    model = bundle['model']
    X = np.array([extract_features(domain, bigram_probs)], dtype=float)
    prob = model.predict_proba(X)[0,1] if hasattr(model,'predict_proba') else model.predict(X)[0]
    label = 'dga' if prob >= 0.5 else 'legit'
    print(f"Domain: {domain}\nModel: {model_type}\nProbability DGA: {prob:.4f}\nPredicted label: {label}")


def parse_args():
    ap = argparse.ArgumentParser(description='Compare RandomForest and XGBoost for DGA detection')
    sub = ap.add_subparsers(dest='command')

    ap_cmp = sub.add_parser('compare', help='Run cross-validation comparison and train final models')
    ap_cmp.add_argument('--csv', default='../随机森林检测DGA/dga_training_data.csv')
    ap_cmp.add_argument('--out', default=MODEL_DIR)
    ap_cmp.add_argument('--rf-est', type=int, default=150)
    ap_cmp.add_argument('--xgb-est', type=int, default=300)
    ap_cmp.add_argument('--k', type=int, default=5)

    ap_pred = sub.add_parser('predict', help='Predict with saved model (rf|xgb)')
    ap_pred.add_argument('--model', choices=['rf','xgb'], required=True)
    ap_pred.add_argument('--domain', required=True)
    ap_pred.add_argument('--model-dir', default=MODEL_DIR)

    return ap.parse_args()


def main():
    args = parse_args()
    if args.command == 'compare':
        comparison(args.csv, args.out, args.rf_est, args.xgb_est, args.k)
    elif args.command == 'predict':
        predict(args.domain, args.model, args.model_dir)
    else:
        print('Specify a command: compare or predict (use -h).')

if __name__ == '__main__':
    main()
