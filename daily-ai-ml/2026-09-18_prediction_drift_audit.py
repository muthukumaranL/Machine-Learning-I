"""Monitor classification prediction drift without requiring labels."""

from __future__ import annotations

import numpy as np


def prediction_drift_audit(
    reference_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compare score distribution and predicted-positive rate across populations."""
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(candidate_scores, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("score arrays must contain finite values")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if np.any((ref < 0) | (ref > 1)) or np.any((cur < 0) | (cur > 1)):
        raise ValueError("classification scores must be probabilities in [0, 1]")

    ref_rate = float(np.mean(ref >= threshold))
    cur_rate = float(np.mean(cur >= threshold))
    return {
        "reference_mean_score": float(np.mean(ref)),
        "candidate_mean_score": float(np.mean(cur)),
        "mean_score_shift": float(np.mean(cur) - np.mean(ref)),
        "reference_positive_rate": ref_rate,
        "candidate_positive_rate": cur_rate,
        "positive_rate_shift": cur_rate - ref_rate,
        "p95_score_shift": float(np.quantile(cur, 0.95) - np.quantile(ref, 0.95)),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    baseline = rng.beta(2, 5, 2000)
    serving = rng.beta(2.4, 4.6, 1200)
    print(prediction_drift_audit(baseline, serving))
