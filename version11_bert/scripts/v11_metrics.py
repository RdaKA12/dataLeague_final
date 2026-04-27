"""Metrics and threshold selection for V11."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def choose_recall_first_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.80,
) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision_M": 0.0, "recall_M": 0.0, "f1_M": 0.0, "f2_M": 0.0}
    precision_t = precision[:-1]
    recall_t = recall[:-1]
    f1 = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-12)
    f2 = 5 * precision_t * recall_t / (4 * precision_t + recall_t + 1e-12)
    candidates = np.where(recall_t >= min_recall)[0]
    if len(candidates):
        idx = int(candidates[np.nanargmax(f1[candidates])])
    else:
        idx = int(np.nanargmax(f2))
    return {
        "threshold": float(thresholds[idx]),
        "precision_M": float(precision_t[idx]),
        "recall_M": float(recall_t[idx]),
        "f1_M": float(f1[idx]),
        "f2_M": float(f2[idx]),
    }


def metric_block(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(set(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_M": float(precision),
        "recall_M": float(recall),
        "f1_M": float(f1),
        "confusion_matrix": {
            "tn_O": int(tn),
            "fp_O_as_M": int(fp),
            "fn_M_as_O": int(fn),
            "tp_M": int(tp),
        },
    }


def review_block(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, margin: float) -> dict[str, Any]:
    review = np.abs(probabilities - threshold) < margin
    decided = ~review
    payload: dict[str, Any] = {
        "review_margin": float(margin),
        "review_rate": float(review.mean()) if len(review) else 0.0,
        "review_n": int(review.sum()),
        "decided_n": int(decided.sum()),
    }
    if decided.sum() and len(set(y_true[decided])) == 2:
        payload["decided_metrics"] = metric_block(y_true[decided], probabilities[decided], threshold)
    return payload

