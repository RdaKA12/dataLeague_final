"""Evaluate V11 on an external labeled CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

import v11_common as common
from v11_inference import predict_many


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V11 on an external M/O labeled CSV")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=common.DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=common.ROOT / "eval_outputs")
    parser.add_argument("--text-col", default="")
    parser.add_argument("--label-col", default="llm_label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def map_label(value: str) -> float:
    normalized = str(value or "").strip().upper()
    if normalized in {"M", "MANIPULATIF", "MANIPULATIVE", "MANIPULAT"}:
        return 1.0
    if normalized in {"O", "ORGANIK", "ORGANIC"}:
        return 0.0
    return np.nan


def choose_text_col(df: pd.DataFrame, explicit: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested text column not found: {explicit}")
        return explicit
    for col in ["clean_text", "llm_text", "original_text", "text"]:
        if col in df.columns:
            return col
    raise ValueError("No text column found. Pass --text-col.")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "actual_counts": {"O": int((y_true == 0).sum()), "M": int((y_true == 1).sum())},
        "predicted_counts": {"O": int((y_pred == 0).sum()), "M": int((y_pred == 1).sum())},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) == 2 else None,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision_M": float(precision),
        "recall_M": float(recall),
        "f1_M": float(f1),
        "confusion_matrix": {
            "layout": "rows=actual, cols=predicted",
            "labels": ["O", "M"],
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tn_O": int(tn),
            "fp_O_as_M": int(fp),
            "fn_M_as_O": int(fn),
            "tp_M": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["O", "M"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    text_col = choose_text_col(df, args.text_col)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    predictions = predict_many(df[text_col].fillna("").astype(str).tolist(), args.model_dir, batch_size=args.batch_size)
    pred_df = pd.DataFrame(predictions)
    scored = pd.concat([df.reset_index(drop=True), pred_df.add_prefix("v11_")], axis=1)
    scored["eval_label_binary"] = scored[args.label_col].map(map_label)
    scored["eval_label_mo"] = scored["eval_label_binary"].map({0.0: "O", 1.0: "M"}).fillna("BELIRSIZ")
    scored["v11_pred_label_mo"] = np.where(scored["v11_manipulative_score"] >= scored["v11_threshold"], "M", "O")

    valid = scored[scored["eval_label_binary"].isin([0.0, 1.0])].copy()
    y_true = valid["eval_label_binary"].astype(int).to_numpy()
    y_prob = valid["v11_manipulative_score"].astype(float).to_numpy()
    threshold = float(scored["v11_threshold"].dropna().iloc[0])
    metrics = {
        "input": str(args.input),
        "model_dir": str(args.model_dir),
        "text_col": text_col,
        "label_col": args.label_col,
        "total_rows": int(len(scored)),
        "evaluated_rows": int(len(valid)),
        "excluded_unmapped_or_belirsiz_rows": int(len(scored) - len(valid)),
        "raw_label_counts": {str(k): int(v) for k, v in scored[args.label_col].value_counts(dropna=False).to_dict().items()},
        "overall": binary_metrics(y_true, y_prob, threshold),
    }

    if "stratum" in valid.columns:
        metrics["by_stratum"] = {}
        for stratum, group in valid.groupby("stratum"):
            if len(group) >= 2 and group["eval_label_binary"].nunique() == 2:
                metrics["by_stratum"][str(stratum)] = binary_metrics(
                    group["eval_label_binary"].astype(int).to_numpy(),
                    group["v11_manipulative_score"].astype(float).to_numpy(),
                    threshold,
                )
            else:
                metrics["by_stratum"][str(stratum)] = {
                    "rows": int(len(group)),
                    "actual_counts": {str(k): int(v) for k, v in group["eval_label_mo"].value_counts().to_dict().items()},
                    "note": "Skipped metrics because only one class is present or too few rows.",
                }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix or args.input.stem
    scored_path = args.output_dir / f"{stem}_v11_scored.csv"
    metrics_path = args.output_dir / f"{stem}_v11_metrics.json"
    confusion_path = args.output_dir / f"{stem}_v11_confusion_matrix.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")
    common.write_json(metrics_path, metrics)
    cm = metrics["overall"]["confusion_matrix"]
    pd.DataFrame(cm["matrix"], index=["actual_O", "actual_M"], columns=["pred_O", "pred_M"]).to_csv(
        confusion_path,
        encoding="utf-8-sig",
    )
    print(
        json.dumps(
            {
                "scored_path": str(scored_path),
                "metrics_path": str(metrics_path),
                "confusion_path": str(confusion_path),
                "summary": metrics["overall"],
                "excluded_unmapped_or_belirsiz_rows": metrics["excluded_unmapped_or_belirsiz_rows"],
            },
            ensure_ascii=False,
            indent=2,
            default=common.json_default,
        )
    )


if __name__ == "__main__":
    main()

