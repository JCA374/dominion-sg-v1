"""The numbers behind the report: AUC, what the inputs already know, partial correlation, time to event."""
from __future__ import annotations

import numpy as np


def auc(score, label) -> float:
    """Mann-Whitney AUC of `score` for the binary `label` (ties averaged). 0.5 = no signal.
    Values below 0.5 mean LOW scores go with the positive class."""
    score, label = np.asarray(score, dtype=float), np.asarray(label).astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_auc(X, y, groups, folds: int = 5, iters: int = 300, lr: float = 0.5, l2: float = 1e-3) -> float:
    """Cross-validated AUC (folds split by episode `groups`) of a logistic regression on X: the reference line
    'what the model's inputs already know'. Plain gradient descent, no dependencies."""
    X, y, groups = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.hstack([X, np.ones((len(X), 1))])
    pred = np.zeros(len(y))
    ug = np.unique(groups)
    folds = min(folds, len(ug))
    for k in range(folds):
        test = np.isin(groups, ug[k::folds])
        Xt, yt = X[~test], y[~test]
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xt @ w))
            w -= lr * (Xt.T @ (p - yt) / len(yt) + l2 * w)
        pred[test] = X[test] @ w
    return auc(pred, y)


def residualize(x, Z):
    """x minus its least-squares fit on Z (with intercept)."""
    Z1 = np.hstack([np.asarray(Z, dtype=float), np.ones((len(Z), 1))])
    beta = np.linalg.lstsq(Z1, np.asarray(x, dtype=float), rcond=None)[0]
    return x - Z1 @ beta


def partial_corr(x, target, Z) -> float:
    """Correlation of x with target after regressing both on Z (the current inputs). Non-zero = NEW information."""
    rx, rt = residualize(x, Z), residualize(target, Z)
    if rx.std() == 0 or rt.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, rt)[0, 1])


def time_to_event(n: int, events) -> np.ndarray:
    """Steps from each index to the next event index (inf after the last event)."""
    nxt = np.full(n, np.inf)
    future = np.inf
    ev = set(int(e) for e in events)
    for i in range(n - 1, -1, -1):
        if i in ev:
            future = i
        nxt[i] = future - i
    return nxt


def leaves_normal(median, lo, hi):
    """Index from which the run-up median stays outside the normal IQR [lo, hi] until the event; None if never."""
    out = (median > hi) | (median < lo)
    idx = len(out)
    while idx > 0 and out[idx - 1]:
        idx -= 1
    return None if idx == len(out) else idx


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a share."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def trailing_mean(x, k: int):
    """Mean of the last k values up to and including each index (shorter at the start)."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)
