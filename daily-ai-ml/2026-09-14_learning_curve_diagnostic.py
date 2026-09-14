"""Diagnose underfitting vs overfitting with a learning curve.

Run:
    python daily-ai-ml/2026-09-14_learning_curve_diagnostic.py

The train/validation gap across sample sizes is more informative than a single
cross-validation score when deciding whether more data or a different model is
likely to help.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve


def learning_curve_audit(model, X: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    """Return train/validation ROC-AUC and generalization gap by sample size."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2D and aligned with y")
    if len(np.unique(y)) != 2:
        raise ValueError("This example expects a binary target")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sizes, train_scores, valid_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    rows: list[dict[str, float]] = []
    for size, train_fold, valid_fold in zip(sizes, train_scores, valid_scores):
        train_mean = float(np.mean(train_fold))
        valid_mean = float(np.mean(valid_fold))
        rows.append(
            {
                "train_samples": float(size),
                "train_auc": train_mean,
                "validation_auc": valid_mean,
                "generalization_gap": train_mean - valid_mean,
                "validation_std": float(np.std(valid_fold)),
            }
        )
    return rows


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=0.9,
        random_state=7,
    )
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)

    print("Learning-curve diagnostic")
    for row in learning_curve_audit(model, X, y):
        print(
            f"n={int(row['train_samples']):4d}  "
            f"train={row['train_auc']:.3f}  valid={row['validation_auc']:.3f}  "
            f"gap={row['generalization_gap']:.3f}  std={row['validation_std']:.3f}"
        )
