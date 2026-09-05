"""Daily AI/ML mini-project: decision-threshold tuning for imbalanced classification.

Concept: a classifier's default 0.50 threshold is not always optimal. This example
chooses the threshold that maximizes F1 on validation-like synthetic data.

Run:
    pip install numpy scikit-learn
    python daily-ai-ml/2026-09-05_threshold_tuning.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scored = [
        (float(threshold), f1_score(y_true, probabilities >= threshold))
        for threshold in thresholds
    ]
    return max(scored, key=lambda item: item[1])


def main() -> None:
    X, y = make_classification(
        n_samples=3000,
        n_features=12,
        n_informative=6,
        weights=[0.90, 0.10],
        flip_y=0.02,
        random_state=42,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)[:, 1]

    best_threshold, best_f1 = find_best_threshold(y_valid, probabilities)
    default_pred = probabilities >= 0.50
    tuned_pred = probabilities >= best_threshold

    print(f"Default threshold: 0.50 | F1={f1_score(y_valid, default_pred):.3f}")
    print(f"Tuned threshold:   {best_threshold:.2f} | F1={best_f1:.3f}")
    print(
        "Tuned precision/recall: "
        f"{precision_score(y_valid, tuned_pred):.3f} / {recall_score(y_valid, tuned_pred):.3f}"
    )


if __name__ == "__main__":
    main()
