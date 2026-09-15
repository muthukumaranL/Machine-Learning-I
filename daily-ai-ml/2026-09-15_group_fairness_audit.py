"""Audit binary classifier performance across demographic or operational groups.

This utility is intentionally model-agnostic: pass labels, probabilities, and a
slice/group column to expose performance gaps hidden by aggregate metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def audit_group_fairness(y_true, y_prob, groups, threshold: float = 0.5) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups)
    if not (len(y_true) == len(y_prob) == len(groups)):
        raise ValueError("Inputs must have equal length")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")

    rows = []
    for group in pd.unique(groups):
        mask = groups == group
        yt, yp = y_true[mask], y_prob[mask]
        pred = (yp >= threshold).astype(int)
        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else np.nan
        rows.append({
            "group": group,
            "n": int(mask.sum()),
            "positive_rate": float(pred.mean()),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "roc_auc": auc,
        })

    result = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    result["recall_gap_from_best"] = result["recall"].max() - result["recall"]
    result["positive_rate_gap_from_best"] = result["positive_rate"].max() - result["positive_rate"]
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(15)
    n = 800
    group = rng.choice(["A", "B", "C"], n, p=[0.45, 0.35, 0.20])
    latent = rng.normal(size=n) + (group == "A") * 0.25 - (group == "C") * 0.2
    truth = (latent + rng.normal(scale=0.8, size=n) > 0).astype(int)
    prob = 1 / (1 + np.exp(-(latent + rng.normal(scale=0.45, size=n))))
    print(audit_group_fairness(truth, prob, group).round(3).to_string(index=False))
