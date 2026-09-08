"""Nested cross-validation for honest model-selection estimates.

Run:
    python daily-ai-ml/2026-09-08_nested_cross_validation.py

The inner loop selects logistic-regression regularization strength while the
outer loop estimates ROC-AUC on data that never influenced hyperparameter
selection. This avoids the optimistic bias of reporting the same CV score used
to choose a model.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def nested_auc(random_state: int = 42) -> tuple[np.ndarray, list[float]]:
    X, y = load_breast_cancer(return_X_y=True)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000)),
        ]
    )
    param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores: list[float] = []
    chosen_c: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state + fold,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner,
            n_jobs=-1,
        )
        search.fit(X[train_idx], y[train_idx])

        probabilities = search.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], probabilities)
        scores.append(float(auc))
        chosen_c.append(float(search.best_params_["model__C"]))
        print(f"fold={fold} auc={auc:.4f} selected_C={chosen_c[-1]:g}")

    return np.asarray(scores), chosen_c


def main() -> None:
    scores, chosen_c = nested_auc()
    print("\nNested CV summary")
    print(f"mean ROC-AUC: {scores.mean():.4f}")
    print(f"std ROC-AUC:  {scores.std(ddof=1):.4f}")
    print(f"selected C values: {chosen_c}")


if __name__ == "__main__":
    main()
