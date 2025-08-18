import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch

from features import extract_features, build_bigram_freq
from lstm_model import DomainLSTM
from train_lstm import encode_domain, clean_domain, _extract_domain_core  # reuse


def load_bigram(model_dir: Path):
    with open(model_dir/'bigram_probs.json','r',encoding='utf-8') as f:
        return json.load(f)

def load_rf_xgb(model_dir: Path, kind: str):
    bundle = joblib.load(model_dir/f'{kind}_model.joblib')
    return bundle['model']

def load_lstm(model_dir: Path, device: str):
    ckpt = torch.load(model_dir/'lstm_model.pt', map_location=device)
    model = DomainLSTM(len(ckpt['vocab']))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, ckpt['vocab'], ckpt['max_len']


def predict_domain(domain: str, model_type: str, dirs, device: str):
    if model_type in ('rf','xgb'):
        model = load_rf_xgb(Path(dirs['classical']), model_type)
        bigram = load_bigram(Path(dirs['classical']))
        feat = extract_features(domain, bigram)
        prob = model.predict_proba([feat])[0,1]
        label = 'dga' if prob>=0.5 else 'legit'
        return label, prob
    elif model_type == 'lstm':
        model, vocab, max_len = load_lstm(Path(dirs['lstm']), device)
        from train_lstm import encode_domain
        ids = encode_domain(domain, vocab, max_len)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()
        label = 'dga' if prob>=0.5 else 'legit'
        return label, prob
    else:
        raise ValueError('Unknown model type')


def loop(args):
    device = args.device
    dirs = {'classical': args.classical_dir, 'lstm': args.lstm_dir}
    model_type = args.model
    samples = [
        '1df5hr42x3s651dgh56tdbq6bs.org',
        '675wwi1hb3y9w1griggr1vxpg33.net',
        'cloud.gist.build',
        'knotch.it',
        'auth.example.com'
    ]
    print(f"Running initial sample predictions with model={model_type}:")
    for d in samples:
        label, prob = predict_domain(d, model_type, dirs, device)
        print(f"{d:40s} -> {label:5s} (p_dga={prob:.4f})")
    print("\nEnter domains (empty line to exit). Model=", model_type)
    print("Type /switch rf|xgb|lstm to change model during session.")
    while True:
        try:
            line = input('> ').strip()
        except EOFError:
            break
        if not line:
            break
        if line.startswith('/switch'):
            parts = line.split()
            if len(parts)==2 and parts[1] in ('rf','xgb','lstm'):
                model_type = parts[1]
                print('Switched model to', model_type)
            else:
                print('Usage: /switch rf|xgb|lstm')
            continue
        label, prob = predict_domain(line, model_type, dirs, device)
        print(f"{line:40s} -> {label:5s} (p_dga={prob:.4f})")


def parse_args():
    ap = argparse.ArgumentParser(description='Interactive predictor for RF/XGB/LSTM models')
    ap.add_argument('--classical-dir', default='artifacts_classical')
    ap.add_argument('--lstm-dir', default='artifacts_lstm')
    ap.add_argument('--model', choices=['rf','xgb','lstm'], default='rf')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    loop(args)
