"""Build distribution-free regression intervals with split conformal prediction.

The calibration set converts absolute residuals into a finite-sample correction
that can wrap any point regressor. This example keeps model training separate so
the interval logic is easy to reuse with scikit-learn, LightGBM, or PyTorch.
"""
from __future__ import annotations

import numpy as np


def conformal_radius(y_calibration, calibration_predictions, alpha: float = 0.1) -> float:
    """Return the split-conformal residual radius for target miscoverage alpha."""
    y = np.asarray(y_calibration, dtype=float)
    pred = np.asarray(calibration_predictions, dtype=float)
    if y.ndim != 1 or y.shape != pred.shape or y.size == 0:
        raise ValueError("calibration targets and predictions must be aligned vectors")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(pred))):
        raise ValueError("calibration inputs must contain only finite values")

    residuals = np.abs(y - pred)
    n = residuals.size
    rank = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.partition(residuals, rank - 1)[rank - 1])


def prediction_intervals(point_predictions, radius: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(point_predictions, dtype=float)
    if pred.ndim != 1 or not np.all(np.isfinite(pred)):
        raise ValueError("point_predictions must be a finite one-dimensional array")
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be a finite non-negative value")
    return pred - radius, pred + radius


def empirical_coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape) or y.ndim != 1 or y.size == 0:
        raise ValueError("y_true, lower, and upper must be aligned vectors")
    return float(np.mean((y >= lo) & (y <= hi)))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_cal = np.linspace(0, 10, 80) + rng.normal(0, 0.8, 80)
    cal_pred = np.linspace(0, 10, 80)
    radius = conformal_radius(y_cal, cal_pred, alpha=0.1)

    test_pred = np.array([2.0, 5.0, 8.0])
    lower, upper = prediction_intervals(test_pred, radius)
    print({"radius": radius, "lower": lower.tolist(), "upper": upper.tolist()})
