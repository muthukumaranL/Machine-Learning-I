"""Audit feature-importance stability across repeated permutation runs.

Run:
    python daily-ai-ml/2026-09-09_permutation_importance_stability.py

The example trains a random forest, computes permutation importance across
multiple seeds, and ranks features by mean importance and coefficient of
variation. Stable importance is stronger evidence than a single ranking.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def audit_stability(repeats: int = 8) -> list[tuple[str, float, float]]:
    X, y = make_classification(
        n_samples=1800,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    runs = []
    for seed in range(repeats):
        result = permutation_importance(
            model, X_test, y_test, scoring="roc_auc", n_repeats=8,
            random_state=seed, n_jobs=-1
        )
        runs.append(result.importances_mean)

    values = np.vstack(runs)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1)
    cv = np.divide(std, np.abs(mean), out=np.full_like(std, np.inf), where=np.abs(mean) > 1e-9)
    rows = [(f"feature_{i}", float(mean[i]), float(cv[i])) for i in range(X.shape[1])]
    return sorted(rows, key=lambda row: row[1], reverse=True)


if __name__ == "__main__":
    print("feature\tmean_importance\tcoefficient_of_variation")
    for name, importance, cv in audit_stability():
        print(f"{name}\t{importance:.4f}\t{cv:.3f}")
