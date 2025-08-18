import argparse
import json
from pathlib import Path
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from features import extract_features, FEATURE_NAMES, build_bigram_freq
from lstm_model import DomainLSTM
from train_lstm import encode_domain, build_vocab, _extract_domain_core, clean_domain  # reuse functions


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    assert {'domain','label'} <= set(df.columns)
    return df


def load_classical(model_dir: Path, kind: str):
    bundle = joblib.load(model_dir / f"{kind}_model.joblib")
    with open(model_dir / 'bigram_probs.json','r',encoding='utf-8') as f:
        bigram_probs = json.load(f)
    return bundle['model'], bigram_probs


def load_lstm(artifact_path: Path, device: str):
    ckpt = torch.load(artifact_path / 'lstm_model.pt', map_location=device)
    vocab = ckpt['vocab']
    max_len = ckpt['max_len']
    model = DomainLSTM(len(vocab))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, vocab, max_len


def evaluate(df: pd.DataFrame, rf_dir: Path, lstm_dir: Path, device: str):
    # Build bigram from legit subset
    bigram_probs = build_bigram_freq(tuple(sorted(set(df[df.label=='legit']['domain']))))
    domains = df['domain'].tolist()
    y_true = (df['label']=='dga').astype(int).values

    # RF
    rf_model, _ = load_classical(rf_dir, 'rf')
    xgb_model, _ = load_classical(rf_dir, 'xgb')

    X = np.array([extract_features(d, bigram_probs) for d in domains])
    rf_prob = rf_model.predict_proba(X)[:,1]
    xgb_prob = xgb_model.predict_proba(X)[:,1]

    # LSTM
    lstm_model, vocab, max_len = load_lstm(lstm_dir, device)
    def encode(d):
        from train_lstm import encode_domain
        return encode_domain(d, vocab, max_len)
    seqs = torch.tensor([encode(d) for d in domains], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = lstm_model(seqs)
        lstm_prob = torch.sigmoid(logits).cpu().numpy()

    results = {}
    for name, prob in [('RandomForest', rf_prob), ('XGBoost', xgb_prob), ('LSTM', lstm_prob)]:
        pred = (prob >= 0.5).astype(int)
        results[name] = {
            'accuracy': accuracy_score(y_true, pred),
            'precision': precision_score(y_true, pred),
            'recall': recall_score(y_true, pred),
            'f1': f1_score(y_true, pred),
            'roc_auc': roc_auc_score(y_true, prob)
        }
    return results


def main():
    ap = argparse.ArgumentParser(description='Compare RF/XGBoost/LSTM on DGA dataset (full evaluation)')
    # default now uses local dataset to eliminate external dependency
    ap.add_argument('--csv', default='dga_training_data.csv')
    ap.add_argument('--classical-dir', default='artifacts_classical')
    ap.add_argument('--lstm-dir', default='artifacts_lstm')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    df = load_dataset(args.csv)
    res = evaluate(df, Path(args.classical_dir), Path(args.lstm_dir), args.device)
    print("=== Model Comparison ===")
    for model, metrics in res.items():
        metric_str = ' | '.join(f"{k}:{v:.4f}" for k,v in metrics.items())
        print(f"{model:12s} -> {metric_str}")

if __name__ == '__main__':
    main()
