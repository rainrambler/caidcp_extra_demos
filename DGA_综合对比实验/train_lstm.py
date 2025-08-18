import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from features import _extract_domain_core, clean_domain
from lstm_model import DomainLSTM

CHARS = list('abcdefghijklmnopqrstuvwxyz0123456789-_.')  # include separators to capture structure
PAD = '<pad>'
UNK = '<unk>'


def build_vocab(domains: List[str]) -> Dict[str, int]:
    vocab = {PAD:0, UNK:1}
    for d in domains:
        core = clean_domain(_extract_domain_core(d))
        for ch in core:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode_domain(domain: str, vocab: Dict[str,int], max_len: int) -> List[int]:
    core = clean_domain(_extract_domain_core(domain))
    ids = [vocab.get(ch, vocab[UNK]) for ch in core[:max_len]]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids

class DomainDataset(Dataset):
    def __init__(self, domains: List[str], labels: List[int], vocab: Dict[str,int], max_len: int):
        self.domains = domains
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.domains)
    def __getitem__(self, idx):
        d = self.domains[idx]
        x = torch.tensor(encode_domain(d, self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    assert {'domain','label'} <= set(df.columns)
    y = (df['label']=='dga').astype(int).tolist()
    return df['domain'].tolist(), y


def split_data(domains: List[str], labels: List[int], test_size: float = 0.2, seed: int = 42):
    random.seed(seed)
    idx = list(range(len(domains)))
    random.shuffle(idx)
    split = int(len(idx)*(1-test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    train_domains = [domains[i] for i in train_idx]
    test_domains = [domains[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_domains, test_domains, train_labels, test_labels


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def eval_epoch(model, loader, device):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            ys.extend(y.cpu().numpy())
            ps.extend(prob.cpu().numpy())
    preds = [1 if p>=0.5 else 0 for p in ps]
    acc = accuracy_score(ys, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(ys, preds, average='binary')
    try:
        auc = roc_auc_score(ys, ps)
    except ValueError:
        auc = float('nan')
    return acc, precision, recall, f1, auc


def train_lstm(csv: str, out: str, max_len: int, batch_size: int, epochs: int, lr: float, embed_dim: int, hidden_dim: int, patience: int, device: str):
    domains, labels = load_data(csv)
    train_domains, test_domains, train_labels, test_labels = split_data(domains, labels)
    vocab = build_vocab(train_domains)

    train_ds = DomainDataset(train_domains, train_labels, vocab, max_len)
    test_ds = DomainDataset(test_domains, test_labels, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = DomainLSTM(len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    patience_left = patience

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc, precision, recall, f1, auc = eval_epoch(model, test_loader, device)
        print(f"Epoch {epoch}: loss={train_loss:.4f} acc={acc:.4f} prec={precision:.4f} recall={recall:.4f} f1={f1:.4f} auc={auc:.4f}")
        improved = f1 > best_f1 + 1e-4
        if improved:
            best_f1 = f1
            patience_left = patience
            # save checkpoint
            out_dir = Path(out)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'max_len': max_len}, out_dir/'lstm_model.pt')
            print("  * Saved new best model")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break
    print(f"Best f1: {best_f1:.4f}")


def parse_args():
    ap = argparse.ArgumentParser(description='Train LSTM model for DGA domain detection')
    # use local dataset copy
    ap.add_argument('--csv', default='dga_training_data.csv')
    ap.add_argument('--out', default='artifacts_lstm')
    ap.add_argument('--max-len', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--embed-dim', type=int, default=64)
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return ap.parse_args()


def main():
    args = parse_args()
    train_lstm(args.csv, args.out, args.max_len, args.batch_size, args.epochs, args.lr, args.embed_dim, args.hidden_dim, args.patience, args.device)

if __name__ == '__main__':
    main()
