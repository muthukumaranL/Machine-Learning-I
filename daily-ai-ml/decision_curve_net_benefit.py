"""Evaluate binary classifiers with decision-curve net benefit.

ROC-AUC measures ranking quality, but deployment decisions happen at a chosen
risk threshold. Decision-curve analysis compares a model with treat-all and
treat-none policies while accounting for the relative cost of false positives.
"""
from __future__ import annotations

import numpy as np


def decision_curve(y_true, y_prob, thresholds=None) -> list[dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and y_prob must be non-empty aligned vectors")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("y_prob must be in [0, 1]")

    thresholds = np.asarray(
        np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds,
        dtype=float,
    )
    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("thresholds must be strictly between 0 and 1")

    n = y.size
    prevalence = float(y.mean())
    rows = []
    for threshold in thresholds:
        predicted = p >= threshold
        tp = int(np.sum(predicted & (y == 1)))
        fp = int(np.sum(predicted & (y == 0)))
        odds = threshold / (1.0 - threshold)
        model_nb = tp / n - fp / n * odds
        all_nb = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "model_net_benefit": float(model_nb),
                "treat_all_net_benefit": float(all_nb),
                "treat_none_net_benefit": 0.0,
                "best_policy": max(
                    ((model_nb, "model"), (all_nb, "treat_all"), (0.0, "treat_none"))
                )[1],
            }
        )
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    labels = rng.binomial(1, 0.25, 500)
    probabilities = np.clip(0.12 + 0.62 * labels + rng.normal(0, 0.18, 500), 0, 1)
    for row in decision_curve(labels, probabilities, [0.1, 0.2, 0.3, 0.4, 0.5]):
        print(row)
