"""Bootstrap confidence interval for ROC-AUC.

A single ROC-AUC number hides sampling uncertainty. This example repeatedly
resamples a held-out prediction set and reports a percentile confidence
interval around the observed AUC.

Run:
    pip install numpy scikit-learn
    python daily-ai-ml/2026-09-07_bootstrap_auc_ci.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return observed AUC and a percentile bootstrap confidence interval."""
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC-AUC requires both classes")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y_true, y_score)
    aucs: list[float] = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) < 2:
            continue
        aucs.append(roc_auc_score(sample_y, y_score[indices]))

    if not aucs:
        raise RuntimeError("no valid bootstrap samples contained both classes")

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(aucs, [alpha, 1 - alpha])
    return float(observed), float(lower), float(upper)


def main() -> None:
    X, y = make_classification(
        n_samples=2500,
        n_features=12,
        n_informative=7,
        weights=[0.80, 0.20],
        class_sep=1.1,
        random_state=7,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=7
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    auc, lower, upper = bootstrap_auc_ci(y_test, scores)
    print(f"ROC-AUC: {auc:.3f}")
    print(f"95% bootstrap CI: [{lower:.3f}, {upper:.3f}]")
    print(f"Interval width: {upper - lower:.3f}")


if __name__ == "__main__":
    main()
