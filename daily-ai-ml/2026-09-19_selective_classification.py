"""Evaluate confidence-based abstention for a probabilistic classifier."""

from __future__ import annotations

import numpy as np


def selective_risk_curve(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: tuple[float, ...] = (0.55, 0.65, 0.75, 0.85, 0.95),
) -> list[dict[str, float]]:
    """Return coverage and error risk as uncertain predictions are rejected."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("y_true and probability must have the same non-empty shape")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must be binary")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite values in [0, 1]")

    confidence = np.maximum(p, 1.0 - p)
    prediction = (p >= 0.5).astype(int)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        if not 0.5 <= threshold <= 1.0:
            raise ValueError("confidence thresholds must be in [0.5, 1]")
        accepted = confidence >= threshold
        n = int(accepted.sum())
        risk = float(np.mean(prediction[accepted] != y[accepted])) if n else float("nan")
        rows.append({
            "threshold": float(threshold),
            "coverage": float(n / y.size),
            "selective_risk": risk,
            "accepted": float(n),
        })
    return rows


if __name__ == "__main__":
    truth = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    scores = np.array([0.08, 0.91, 0.58, 0.42, 0.81, 0.30, 0.52, 0.12])
    for row in selective_risk_curve(truth, scores):
        print(row)
