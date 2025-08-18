"""Quick interactive test for the trained DGA RF model.
Run after training:
    python quick_test.py --model-dir model_artifacts
You can also pass domains by --domains or type them interactively.
"""
import argparse
from train_dga_rf import load_model
from features import extract_features
import numpy as np

SAMPLE_DOMAINS = [
    # suspected dga-like
    "1df5hr42x3s651dgh56tdbq6bs.org",
    "675wwi1hb3y9w1griggr1vxpg33.net",
    # legit-like
    "cloud.gist.build",
    "knotch.it",
    "auth.example.com",
]

def predict_multi(domains, model_dir):
    clf, bigram_probs, feat_names = load_model(model_dir)
    feats = [extract_features(d, bigram_probs) for d in domains]
    X = np.array(feats, dtype=float)
    probs = clf.predict_proba(X)[:,1]
    for d,p in zip(domains, probs):
        label = 'dga' if p >= 0.5 else 'legit'
        print(f"{d:40s} -> {label:5s} (p_dga={p:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='model_artifacts')
    ap.add_argument('--domains', nargs='*', help='Optional list of domains to test')
    args = ap.parse_args()

    domains = args.domains if args.domains else SAMPLE_DOMAINS
    predict_multi(domains, args.model_dir)

    if not args.domains:
        print("\nEnter domains line-by-line (empty line to exit):")
        while True:
            try:
                d = input('> ').strip()
            except EOFError:
                break
            if not d:
                break
            predict_multi([d], args.model_dir)

if __name__ == '__main__':
    main()
