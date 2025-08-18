import argparse
import json
from pathlib import Path
import joblib
import numpy as np

from train_compare import extract_features  # reuse


def load_artifacts(model_dir: str, model_type: str):
    with open(Path(model_dir)/'bigram_probs.json','r',encoding='utf-8') as f:
        bigram_probs = json.load(f)
    model_bundle = joblib.load(Path(model_dir)/f'{model_type}_model.joblib')
    return model_bundle['model'], bigram_probs, model_bundle['feature_names']


def predict(domains, model_dir, model_type):
    model, bigram_probs, feat_names = load_artifacts(model_dir, model_type)
    from train_compare import extract_features  # ensure function available
    feats = [extract_features(d, bigram_probs) for d in domains]
    X = np.array(feats, dtype=float)
    probs = model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.predict(X)
    for d,p in zip(domains, probs):
        label = 'dga' if p >= 0.5 else 'legit'
        print(f"{d:40s} -> {label:5s} (p_dga={p:.4f})")


def main():
    ap = argparse.ArgumentParser(description='Quick predict using trained RF/XGB models')
    ap.add_argument('--model-dir', default='model_artifacts')
    ap.add_argument('--model', choices=['rf','xgb'], default='xgb')
    ap.add_argument('--domains', nargs='*', help='Domains to classify')
    args = ap.parse_args()

    test_domains = args.domains or [
        '1df5hr42x3s651dgh56tdbq6bs.org',
        '675wwi1hb3y9w1griggr1vxpg33.net',
        'cloud.gist.build',
        'knotch.it',
        'auth.example.com'
    ]
    predict(test_domains, args.model_dir, args.model)

if __name__ == '__main__':
    main()
