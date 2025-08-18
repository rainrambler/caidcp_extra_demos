import math
import re
from collections import Counter
from functools import lru_cache
from typing import List, Tuple, Dict

VOWELS = set("aeiou")
WORD_DICT = set([
    # A tiny built-in word list (can be extended). Keep small for demo.
    "mail","cloud","data","shop","news","play","game","app","video","music","chat","safe","secure","update","service","cdn","api","prod","dev","beta","test","edge","node","live","time","zoom","fast","net","web","log","track","ad","ads","img","static","server","auth","core","global","world","home","main","user","social","media","file","upload","download","store","search","map","drive","link","panel","admin","portal","config","sync","cache"
])

DOMAIN_CHARS_RE = re.compile(r"[a-z0-9]")


def _extract_domain_core(domain: str) -> str:
    """Return the registrable portion (very rough heuristic): last two labels of domain before any path."""
    domain = domain.strip().lower().split('/')[0]
    parts = [p for p in domain.split('.') if p]
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else domain


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((freq / n) * math.log2(freq / n) for freq in c.values())


def vowel_count(s: str) -> int:
    return sum(ch in VOWELS for ch in s)


def digit_count(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def repeated_char_count(s: str) -> int:
    if not s:
        return 0
    counts = Counter(s)
    return sum(freq for freq in counts.values() if freq > 1)


def max_consecutive_digits(s: str) -> int:
    max_run = run = 0
    for ch in s:
        if ch.isdigit():
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def max_consecutive_consonants(s: str) -> int:
    max_run = run = 0
    for ch in s:
        if ch.isalpha() and ch not in VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def unique_char_count(s: str) -> int:
    return len(set(s))


def clean_domain(domain: str) -> str:
    # keep only a-z0-9 for modeling core features
    domain = domain.lower()
    return ''.join(ch for ch in domain if ch.isalnum())


def char_bigrams(s: str) -> List[str]:
    return [s[i:i+2] for i in range(len(s) - 1)] if len(s) > 1 else []

@lru_cache(maxsize=2048)
def build_bigram_freq(domains: Tuple[str, ...]) -> Dict[str, float]:
    counts = Counter()
    total = 0
    for d in domains:
        core = clean_domain(_extract_domain_core(d))
        for bg in char_bigrams(core):
            counts[bg] += 1
            total += 1
    if total == 0:
        return {}
    # Convert to probabilities with Laplace smoothing across observed set size
    V = len(counts)
    return {bg: (c + 1) / (total + V) for bg, c in counts.items()}


def bigram_score(domain: str, bigram_probs: Dict[str, float]) -> float:
    # average log probability of bigrams; unseen bigram uses minimal smoothing
    core = clean_domain(_extract_domain_core(domain))
    bgs = char_bigrams(core)
    if not bgs:
        return 0.0
    default = 1.0 / (sum(bigram_probs.values()) + len(bigram_probs) or 1)
    return sum(math.log(bigram_probs.get(bg, default)) for bg in bgs) / len(bgs)


def dictionary_word_coverage(domain: str) -> float:
    # naive segmentation: sliding longest word matches over the core domain
    core = clean_domain(_extract_domain_core(domain))
    if not core:
        return 0.0
    matched = [False] * len(core)
    # attempt to match dictionary words longest-first
    words_sorted = sorted(WORD_DICT, key=len, reverse=True)
    for w in words_sorted:
        start = 0
        while True:
            idx = core.find(w, start)
            if idx == -1:
                break
            for i in range(idx, idx + len(w)):
                matched[i] = True
            start = idx + 1
    return sum(matched) / len(core)


def extract_features(domain: str, bigram_probs: Dict[str, float]) -> List[float]:
    core = clean_domain(_extract_domain_core(domain))
    length = len(core)
    ent = shannon_entropy(core)
    vowels = vowel_count(core)
    digits = digit_count(core)
    repeated = repeated_char_count(core)
    max_dig = max_consecutive_digits(core)
    max_cons = max_consecutive_consonants(core)
    uniq = unique_char_count(core)
    vowel_ratio = vowels / length if length else 0
    digit_ratio = digits / length if length else 0
    bigram_sc = bigram_score(domain, bigram_probs)
    dict_cov = dictionary_word_coverage(domain)
    return [
        length, ent, vowels, digits, repeated, max_dig, max_cons, uniq,
        vowel_ratio, digit_ratio, bigram_sc, dict_cov
    ]

FEATURE_NAMES = [
    "length","entropy","vowel_count","digit_count","repeated_char_count","max_consecutive_digits",
    "max_consecutive_consonants","unique_char_count","vowel_ratio","digit_ratio","bigram_avg_logp","dict_coverage"
]

__all__ = [
    "extract_features","FEATURE_NAMES","build_bigram_freq"
]
