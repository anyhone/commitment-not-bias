"""BAD classifier (Biased Activation Detection). Label: 1 = biased, 0 = unbiased."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    p = Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000))])
    p.fit(X.astype(np.float32), y.astype(int))
    return p


def to_torch(p: Pipeline, device="cuda") -> dict:
    sc, clf = p.named_steps["scaler"], p.named_steps["clf"]
    assert list(clf.classes_) == [0, 1]
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=device)
    return {"mean": t(sc.mean_), "scale": t(sc.scale_), "w": t(clf.coef_[0]), "b": t(clf.intercept_[0])}


def balanced_acc(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    r = [(yhat[y == c] == c).mean() for c in [0, 1] if (y == c).any()]
    return float(np.mean(r) * 100)
