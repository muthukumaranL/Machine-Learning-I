"""Compute Expected Calibration Error (ECE) for binary classifiers.

This small utility complements ranking metrics such as ROC-AUC by checking
whether predicted probabilities match observed event frequencies.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional and aligned")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        mask = ids == b
        if not np.any(mask):
            continue
        confidence = float(p[mask].mean())
        accuracy = float(y[mask].mean())
        weight = float(mask.mean())
        gap = abs(accuracy - confidence)
        ece += weight * gap
        rows.append({"bin": b, "count": int(mask.sum()), "confidence": confidence,
                     "observed_rate": accuracy, "gap": gap})
    return {"ece": float(ece), "bins": rows}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    probabilities = rng.uniform(0.05, 0.95, 1000)
    outcomes = rng.binomial(1, probabilities)
    result = expected_calibration_error(outcomes, probabilities)
    print(f"ECE: {result['ece']:.4f}")
    for row in result["bins"]:
        print(row)
