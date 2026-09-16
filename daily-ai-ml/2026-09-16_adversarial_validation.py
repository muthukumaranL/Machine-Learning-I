"""Adversarial validation for detecting train/serving distribution shift.

A classifier is trained to distinguish reference rows from candidate rows.
ROC-AUC near 0.5 suggests similar distributions; higher AUC indicates that
features contain enough signal to identify which dataset a row came from.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def adversarial_validation_auc(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Return cross-validated adversarial AUC and an interpretable shift score."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference and candidate must be 2D arrays")
    if reference.shape[1] != candidate.shape[1]:
        raise ValueError("datasets must have the same number of features")

    X = np.vstack([reference, candidate])
    y = np.concatenate([np.zeros(len(reference)), np.ones(len(candidate))])
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    cv = StratifiedKFold(cv_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for train_idx, test_idx in cv.split(X, y):
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        scores.append(roc_auc_score(y[test_idx], fitted.predict_proba(X[test_idx])[:, 1]))

    auc = float(np.mean(scores))
    return {
        "adversarial_auc": auc,
        "auc_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "shift_score": float(max(0.0, (auc - 0.5) * 2.0)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    train = rng.normal(0.0, 1.0, size=(800, 6))
    serving = rng.normal(0.35, 1.0, size=(500, 6))
    print(adversarial_validation_auc(train, serving))
