"""Population Stability Index (PSI) for feature drift monitoring."""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> dict[str, float | list[float]]:
    """Compare one numeric feature across reference and candidate populations."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(candidate, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and candidate must contain finite values")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        raise ValueError("reference feature needs enough variation for PSI")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    contributions = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi = float(contributions.sum())

    return {
        "psi": psi,
        "drift_level": "high" if psi >= 0.25 else "moderate" if psi >= 0.10 else "low",
        "bin_contributions": contributions.tolist(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    baseline = rng.normal(0.0, 1.0, 2000)
    shifted = rng.normal(0.4, 1.1, 1200)
    print(population_stability_index(baseline, shifted))
