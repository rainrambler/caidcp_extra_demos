import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import joblib
import numpy as np
import warnings
from typing import List, Dict

from lstm_model import DomainLSTM
from features import extract_features, FEATURE_NAMES, build_bigram_freq, clean_domain, _extract_domain_core

# Utilities reused from training scripts

def load_classical_artifacts(classical_dir: str):
    cdir = Path(classical_dir)
    rf_path = cdir / 'rf_model.joblib'
    xgb_path = cdir / 'xgb_model.joblib'
    bigram_path = cdir / 'bigram_probs.json'
    artifacts = {}
    if rf_path.exists():
        artifacts['rf'] = joblib.load(rf_path)
    if xgb_path.exists():
        artifacts['xgb'] = joblib.load(xgb_path)
    if bigram_path.exists():
        import json
        with open(bigram_path, 'r', encoding='utf-8') as f:
            artifacts['bigram_probs'] = json.load(f)
    return artifacts

def load_lstm_artifacts(lstm_dir: str, device: str):
    ldir = Path(lstm_dir)
    ckpt_path = ldir / 'lstm_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt['vocab']
    max_len = ckpt['max_len']
    model = DomainLSTM(len(vocab))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, vocab, max_len

def encode_domain_for_vocab(domain: str, vocab: Dict[str,int], max_len: int):
    core = clean_domain(_extract_domain_core(domain))
    ids = [vocab.get(ch, vocab['<unk>']) for ch in core[:max_len]]
    if len(ids) < max_len:
        ids += [vocab['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def feature_importance_contrib(model_bundle, domain: str, bigram_probs: Dict[str,float]):
    model = model_bundle['model'] if 'model' in model_bundle else model_bundle
    # Some joblib saved as dict? Support direct estimator
    feats = extract_features(domain, bigram_probs)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        contribs = [(n, f, imp, f*imp) for n,f,imp in zip(FEATURE_NAMES, feats, importances)]
        contribs.sort(key=lambda x: abs(x[3]), reverse=True)
        return contribs
    return []


def lstm_token_saliency(model: DomainLSTM, x_ids: torch.Tensor, target_label: int, device: str):
    """Compute gradient saliency per token & hidden state norms.
    Uses a single forward pass with embeddings retaining grad to avoid cudnn eval-mode backward issues.
    """
    x_ids = x_ids.to(device)
    was_training = model.training
    # Enable training mode so cudnn RNN stores intermediates; disable dropout effects.
    model.train()
    dropout_mode = model.dropout.training
    model.dropout.eval()
    try:
        emb = model.embedding(x_ids)
        emb.retain_grad()
        lstm_out, (h, c) = model.lstm(emb)
        if model.lstm.bidirectional:
            h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        else:
            h_cat = h[-1]
        logits = model.fc(h_cat).squeeze(-1)
        prob = torch.sigmoid(logits).item()
        model.zero_grad()
        logits.backward()
        grad = emb.grad[0]
        token_importance = grad.norm(dim=1).detach()
        hidden_norm = lstm_out[0].detach().norm(dim=1)
    except RuntimeError as e:
        if 'cudnn' in str(e).lower() and device.startswith('cuda'):
            print('[Warn] CUDA backward failed, retrying on CPU for saliency...')
            cpu_model = model.to('cpu')
            return lstm_token_saliency(cpu_model, x_ids.to('cpu'), target_label, 'cpu')
        raise
    finally:
        if dropout_mode:
            model.dropout.train()
        if not was_training:
            model.eval()
        if device != str(next(model.parameters()).device):
            model.to(device)
    return prob, token_importance.cpu().numpy(), hidden_norm.cpu().numpy()


def visualize_saliency(domain: str, vocab: Dict[str,int], token_importance: np.ndarray, hidden_norm: np.ndarray, max_len: int):
    core = clean_domain(_extract_domain_core(domain))[:max_len]
    chars = list(core)
    # Align lengths
    L = min(len(chars), len(token_importance))
    rows = []
    max_imp = token_importance[:L].max() if L>0 else 1.0
    max_h = hidden_norm[:L].max() if L>0 else 1.0
    for i in range(L):
        ch = chars[i]
        imp = token_importance[i]
        hn = hidden_norm[i]
        bar_imp = '█'*int(imp/max_imp*10) if max_imp>0 else ''
        bar_h = '▓'*int(hn/max_h*10) if max_h>0 else ''
        rows.append(f"{i:02d} {ch}  grad:{imp:6.4f} {bar_imp:<10}  hid:{hn:6.4f} {bar_h:<10}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description='Explain predictions: classical feature importance & LSTM token saliency')
    ap.add_argument('--classical-dir', default='artifacts_classical')
    ap.add_argument('--lstm-dir', default='artifacts_lstm')
    ap.add_argument('--domain', required=True, help='Domain string to explain')
    ap.add_argument('--model', default='rf', choices=['rf','xgb'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no-warmup', action='store_true', help='Disable CUDA warm-up (may show cuBLAS context warning)')
    args = ap.parse_args()

    # Device sanity & warm-up to avoid cuBLAS primary context warning
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('[Info] CUDA not available. Falling back to CPU.')
            args.device = 'cpu'
        elif not args.no_warmup:
            try:
                torch.cuda.init()
                # Minimal tensor alloc to create primary context
                _ = torch.zeros(1, device=args.device)
                # Explicit cuBLAS handle warm-up via a tiny matmul (guards later LSTM backward)
                a = torch.randn(32, 32, device=args.device)
                b = torch.randn(32, 32, device=args.device)
                _ = torch.mm(a, b)
                del a, b
                torch.cuda.synchronize()
            except Exception as e:
                print(f'[Warn] CUDA warm-up failed ({e}), falling back to CPU for explanation.')
                args.device = 'cpu'

    # Load artifacts
    classical = load_classical_artifacts(args.classical_dir)
    if 'bigram_probs' not in classical:
        print('Bigram probabilities not found, classical explanation limited.')
        bigram_probs = {}
    else:
        bigram_probs = classical['bigram_probs']

    # Classical feature importance
    if args.model not in classical:
        print(f"Classical model {args.model} not found. Available: {list(classical.keys())}")
    else:
        contribs = feature_importance_contrib(classical[args.model], args.domain, bigram_probs)
        if contribs:
            print("=== Classical Feature Contributions (sorted by |value*importance|) ===")
            print(f"{'Feature':<30}{'Value':>10}{'Imp':>10}{'Value*Imp':>12}")
            for name,val,imp,prod in contribs:
                print(f"{name:<30}{val:10.4f}{imp:10.4f}{prod:12.4f}")
        else:
            print('No feature importances available for this model.')

    # LSTM saliency
    model, vocab, max_len = load_lstm_artifacts(args.lstm_dir, args.device)
    # Post-load warm-up (cuBLAS/cudnn handles) to further reduce first-backward warning
    if args.device.startswith('cuda') and not args.no_warmup:
        try:
            # Suppress expected first-time cuBLAS/cudnn warnings during warm-up backward
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*no current CUDA context.*')
                model.train(); model.dropout.eval()
                dummy_ids = torch.zeros(1, max_len, dtype=torch.long, device=args.device)
                emb = model.embedding(dummy_ids); emb.retain_grad()
                lstm_out, (h, c) = model.lstm(emb)
                if model.lstm.bidirectional:
                    h_cat = torch.cat([h[-2], h[-1]], dim=-1)
                else:
                    h_cat = h[-1]
                logit = model.fc(h_cat).squeeze(-1)
                logit.backward(); model.zero_grad(); model.eval()
        except Exception as e:
            print(f"[Warn] CUDA post-load warm-up failed ({e}); continuing...")
    x_ids = encode_domain_for_vocab(args.domain, vocab, max_len)
    prob, token_imp, hidden_norm = lstm_token_saliency(model, x_ids, target_label=1, device=args.device)
    print("\n=== LSTM Token Saliency & Hidden State Norms ===")
    print(f"Predicted p(dga)={prob:.4f}")
    print(visualize_saliency(args.domain, vocab, token_imp, hidden_norm, max_len))

if __name__ == '__main__':
    main()
