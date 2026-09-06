"""Daily AI/ML mini-project — Population Stability Index (PSI).

PSI is a lightweight production-ML signal for detecting distribution shift
between a reference population (for example, training data) and a newer batch.
This example implements the calculation from scratch with NumPy and shows how
binning and zero-count protection affect a practical drift monitor.

Run:
    pip install numpy
    python daily-ai-ml/2026-09-06_population_stability_index.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-6


@dataclass(frozen=True)
class PSIResult:
    score: float
    reference_share: np.ndarray
    current_share: np.ndarray
    bin_edges: np.ndarray

    @property
    def interpretation(self) -> str:
        if self.score < 0.10:
            return "little distribution change"
        if self.score < 0.25:
            return "moderate distribution change"
        return "substantial distribution change"


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> PSIResult:
    """Compare two numeric distributions using quantile bins from reference data."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("reference and current must be one-dimensional")
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current must not be empty")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if not (np.isfinite(reference).all() and np.isfinite(current).all()):
        raise ValueError("inputs must contain only finite values")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("reference data does not have enough unique values for PSI")

    # Extend edge bins so new production values outside the training range are counted.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = reference_counts / reference_counts.sum()
    current_share = current_counts / current_counts.sum()

    reference_safe = np.clip(reference_share, EPSILON, None)
    current_safe = np.clip(current_share, EPSILON, None)
    score = float(
        np.sum((current_safe - reference_safe) * np.log(current_safe / reference_safe))
    )

    return PSIResult(
        score=score,
        reference_share=reference_share,
        current_share=current_share,
        bin_edges=edges,
    )


def main() -> None:
    rng = np.random.default_rng(42)
    training_feature = rng.normal(loc=50.0, scale=10.0, size=5_000)

    stable_batch = rng.normal(loc=50.5, scale=10.1, size=2_000)
    shifted_batch = rng.normal(loc=57.0, scale=12.0, size=2_000)

    for name, batch in [("stable", stable_batch), ("shifted", shifted_batch)]:
        result = population_stability_index(training_feature, batch)
        print(
            f"{name:7s} PSI={result.score:.4f} -> {result.interpretation}"
        )


if __name__ == "__main__":
    main()
