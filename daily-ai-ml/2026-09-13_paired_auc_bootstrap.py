"""Compare two classifiers with a paired bootstrap confidence interval.

Run:
    python daily-ai-ml/2026-09-13_paired_auc_bootstrap.py

Using the same bootstrap samples for both models preserves the pairing between
predictions and gives a confidence interval for the ROC-AUC difference itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def paired_auc_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (len(y_true) == len(score_a) == len(score_b)):
        raise ValueError("y_true and both score arrays must have equal length")
    if len(np.unique(y_true)) != 2:
        raise ValueError("ROC-AUC requires a binary target with both classes present")

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_y = y_true[idx]
        if len(np.unique(sampled_y)) < 2:
            continue
        auc_a = roc_auc_score(sampled_y, score_a[idx])
        auc_b = roc_auc_score(sampled_y, score_b[idx])
        differences.append(float(auc_a - auc_b))

    if not differences:
        raise RuntimeError("No valid bootstrap samples contained both classes")

    alpha = 1.0 - confidence
    low, high = np.quantile(differences, [alpha / 2, 1 - alpha / 2])
    observed_a = float(roc_auc_score(y_true, score_a))
    observed_b = float(roc_auc_score(y_true, score_b))
    return {
        "auc_a": observed_a,
        "auc_b": observed_b,
        "difference": observed_a - observed_b,
        "ci_low": float(low),
        "ci_high": float(high),
        "valid_bootstrap_samples": float(len(differences)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.35, size=600)
    signal = y + rng.normal(0, 0.9, size=len(y))
    model_a = signal + rng.normal(0, 0.25, size=len(y))
    model_b = signal + rng.normal(0, 0.55, size=len(y))

    result = paired_auc_bootstrap(y, model_a, model_b)
    print("Paired ROC-AUC comparison")
    for key, value in result.items():
        print(f"{key:>24}: {value:.4f}")
