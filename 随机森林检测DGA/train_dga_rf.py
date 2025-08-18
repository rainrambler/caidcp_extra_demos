import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from features import extract_features, FEATURE_NAMES, build_bigram_freq

MODEL_META_NAME = "dga_rf_model.joblib"
BIGRAM_META_NAME = "bigram_probs.json"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity
    assert {'domain','label'} <= set(df.columns), "CSV must contain domain,label columns"
    return df


def build_bigram_model(legit_domains: List[str]):
    # Use only unique legit domains for bigram probability building to reduce bias
    unique_legit = tuple(sorted(set(legit_domains)))
    return build_bigram_freq(unique_legit)


def vectorize(domains: List[str], bigram_probs: Dict[str, float]) -> np.ndarray:
    feats = [extract_features(d, bigram_probs) for d in domains]
    return np.array(feats, dtype=float)


def train(csv_path: str, out_dir: str, test_size: float = 0.2, random_state: int = 42, n_estimators: int = 300):
    df = load_data(csv_path)
    bigram_probs = build_bigram_model(df[df.label == 'legit']['domain'].tolist())

    X = vectorize(df['domain'].tolist(), bigram_probs)
    y = (df['label'] == 'dga').astype(int).values

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X, y, df['domain'].tolist(), test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:,1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    print("=== Evaluation Metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['legit','dga']))

    os.makedirs(out_dir, exist_ok=True)
    model_path = Path(out_dir) / MODEL_META_NAME
    joblib.dump({
        'model': clf,
        'feature_names': FEATURE_NAMES
    }, model_path)
    print(f"Saved model to {model_path}")

    bigram_path = Path(out_dir) / BIGRAM_META_NAME
    with open(bigram_path, 'w', encoding='utf-8') as f:
        json.dump(bigram_probs, f)
    print(f"Saved bigram probabilities to {bigram_path}")


def load_model(model_dir: str):
    bundle = joblib.load(Path(model_dir) / MODEL_META_NAME)
    with open(Path(model_dir) / BIGRAM_META_NAME, 'r', encoding='utf-8') as f:
        bigram_probs = json.load(f)
    return bundle['model'], bigram_probs, bundle['feature_names']


def predict_domain(domain: str, model_dir: str):
    clf, bigram_probs, feat_names = load_model(model_dir)
    X = np.array([extract_features(domain, bigram_probs)], dtype=float)
    prob = clf.predict_proba(X)[0,1]
    label = 'dga' if prob >= 0.5 else 'legit'
    print(f"Domain: {domain}\nPredicted label: {label}\nProbability DGA: {prob:.4f}")
    # Also show feature contributions (via feature importances * value as crude heuristic)
    contrib = {feat: float(val) * float(imp) for feat, val, imp in zip(feat_names, X[0], clf.feature_importances_)}
    top = sorted(contrib.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print("Top feature signals (heuristic):")
    for k,v in top:
        print(f"  {k}: {v:.4f} (value={X[0][feat_names.index(k)]:.4f}, importance={clf.feature_importances_[feat_names.index(k)]:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description="Random Forest DGA domain detection demo")
    sub = ap.add_subparsers(dest='command')

    ap_train = sub.add_parser('train', help='Train model')
    ap_train.add_argument('--csv', default='dga_training_data.csv', help='CSV path')
    ap_train.add_argument('--out', default='model_artifacts', help='Output directory')
    ap_train.add_argument('--test-size', type=float, default=0.2)
    ap_train.add_argument('--n-estimators', type=int, default=300)

    ap_pred = sub.add_parser('predict', help='Predict single domain')
    ap_pred.add_argument('--domain', required=True)
    ap_pred.add_argument('--model-dir', default='model_artifacts')

    return ap.parse_args()


def main():
    args = parse_args()
    if args.command == 'train':
        train(args.csv, args.out, test_size=args.test_size, n_estimators=args.n_estimators)
    elif args.command == 'predict':
        predict_domain(args.domain, args.model_dir)
    else:
        print("Please specify a command: train or predict. Use -h for help.")

if __name__ == '__main__':
    main()
