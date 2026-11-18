"""Splits, metrics and benchmarks.

Label order is H=0, D=1, A=2 everywhere. Probability arrays are n x 3 in that order.
Owner: Cheypol
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

WARMUP = ["0506", "0607", "0708", "0809", "0910"]       # never trained on
TRAIN = ["1011", "1112", "1213", "1314", "1415", "1516",
         "1617", "1718", "1819", "1920", "2021", "2122"]
VAL = ["2223"]
TEST = ["2324", "2425"]                                 # touch ONCE, end of week 3


def split(df):
    """Return train, val, test by season (never random)."""
    return (df[df.season.isin(TRAIN)], df[df.season.isin(VAL)], df[df.season.isin(TEST)])


# ---- metrics -----------------------------------------------------------------------------
def rps(y, P):
    """Ranked Probability Score (lower is better). Perfect prediction = 0."""
    y, P = np.asarray(y, dtype=int), np.asarray(P, dtype=float)
    O = np.eye(3)[y]
    cp, co = np.cumsum(P, axis=1), np.cumsum(O, axis=1)
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2))


def brier(y, P):
    """Multi-class Brier score (lower is better). Range 0 to 2."""
    y, P = np.asarray(y, dtype=int), np.asarray(P, dtype=float)
    return float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))


def evaluate(y, P):
    y, P = np.asarray(y, dtype=int), np.asarray(P, dtype=float)
    pred = P.argmax(axis=1)
    return {"accuracy": round(accuracy_score(y, pred), 4),
            "macro_F1": round(f1_score(y, pred, average="macro"), 4),
            "brier": round(brier(y, P), 4),
            "rps": round(rps(y, P), 4)}


# ---- benchmarks ------------------------------------------------------------------------
def always_home(n):
    """Predict a home win with certainty. Floor for accuracy; harsh on Brier/RPS."""
    return np.tile([1.0, 0.0, 0.0], (n, 1))


def frequencies(train_labels, n):
    """Same probabilities for every match = how often H/D/A happened in training seasons."""
    freq = np.bincount(np.asarray(train_labels, dtype=int), minlength=3) / len(train_labels)
    return np.tile(freq, (n, 1))


def bookmaker_probs(df):
    """Bet365 odds -> probabilities, with the bookmaker margin (overround) removed."""
    inv = 1 / df[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def benchmark_table(df):
    """Rows for the Results Log: the 3 benchmarks on validation and test."""
    import pandas as pd
    train, val, test = split(df)
    rows = []
    for name, part in [("val", val), ("test", test)]:
        rows.append({"model": "always home", "split": name, **evaluate(part.label, always_home(len(part)))})
        rows.append({"model": "frequencies", "split": name, **evaluate(part.label, frequencies(train.label, len(part)))})
        ok = part.dropna(subset=["odds_h", "odds_d", "odds_a"])
        rows.append({"model": "bookmaker (B365)", "split": name, **evaluate(ok.label, bookmaker_probs(ok))})
    return pd.DataFrame(rows)
