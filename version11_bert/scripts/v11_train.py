"""Train the V11 text-only multilingual BERT classifier."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import v11_common as common
from v11_metrics import choose_recall_first_threshold, metric_block, review_block


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray | None = None) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, object]:
        item: dict[str, object] = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V11 BERT text-only MO classifier")
    parser.add_argument("--labels", type=Path, default=common.DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=common.DEFAULT_PROCESSED)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--review-margin", type=float, default=0.08)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-tfidf", action="store_true")
    return parser.parse_args()


def make_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = df.groupby("group_key", as_index=False)["target_m"].max()
    trainval_groups, holdout_groups = train_test_split(
        groups,
        test_size=0.15,
        random_state=seed,
        stratify=groups["target_m"],
    )
    train_groups, val_groups = train_test_split(
        trainval_groups,
        test_size=0.15 / 0.85,
        random_state=seed + 1,
        stratify=trainval_groups["target_m"],
    )
    split_map = {g: "train" for g in train_groups["group_key"]}
    split_map.update({g: "val" for g in val_groups["group_key"]})
    split_map.update({g: "holdout" for g in holdout_groups["group_key"]})
    out = df.copy()
    out["split"] = out["group_key"].map(split_map)
    return out


def make_collate_fn(tokenizer: Any, max_length: int):
    def collate(rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts = [str(row["text"]) for row in rows]
        batch = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "labels" in rows[0]:
            batch["labels"] = torch.tensor([int(row["labels"]) for row in rows], dtype=torch.long)
        return batch

    return collate


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(**batch).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs)


def train_tfidf(train_df: pd.DataFrame, val_df: pd.DataFrame, holdout_df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=300_000,
        lowercase=True,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_holdout = vectorizer.transform(holdout_df["text"])
    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
        solver="saga",
        random_state=59,
    )
    clf.fit(X_train, train_df["target_m"].to_numpy())
    val_prob = clf.predict_proba(X_val)[:, 1]
    threshold_info = choose_recall_first_threshold(val_df["target_m"].to_numpy(), val_prob)
    holdout_prob = clf.predict_proba(X_holdout)[:, 1]
    metrics = {
        "validation": metric_block(val_df["target_m"].to_numpy(), val_prob, threshold_info["threshold"]),
        "holdout": metric_block(holdout_df["target_m"].to_numpy(), holdout_prob, threshold_info["threshold"]),
    }

    examples_df = pd.concat([train_df, val_df], ignore_index=True)
    example_cols = [
        "source_row_id",
        "text",
        "human_label",
        "labeler_notes",
        "language",
        "primary_theme",
        "url",
    ]
    for col in example_cols:
        if col not in examples_df.columns:
            examples_df[col] = ""
    X_examples = vectorizer.transform(examples_df["text"])
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
    nn.fit(X_examples)
    nearest_bundle = {
        "vectorizer": vectorizer,
        "matrix": X_examples,
        "neighbors": nn,
        "examples": examples_df[example_cols + ["target_m"]].to_dict(orient="records"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf, "threshold": threshold_info["threshold"], "metrics": metrics}, output_dir / "tfidf_baseline.joblib")
    joblib.dump(nearest_bundle, output_dir / "nearest_examples.joblib")
    return {"metrics": metrics, "threshold": threshold_info["threshold"], "validation_prob": val_prob, "holdout_prob": holdout_prob}


def train() -> dict[str, Any]:
    args = parse_args()
    common.set_seed(args.seed)
    output_dir = args.output_dir
    model_dir = output_dir / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        raise SystemExit(
            "CUDA is not available. Install CUDA-enabled PyTorch first, or pass --allow-cpu only for smoke tests."
        )
    device = torch.device("cuda" if cuda_available else "cpu")
    use_amp = device.type == "cuda"

    df = common.load_labels(args.labels)
    if args.sample_size and args.sample_size < len(df):
        sampled_parts = []
        for _, group in df.groupby("human_label", sort=False):
            n = max(1, int(round(args.sample_size * len(group) / len(df))))
            sampled_parts.append(group.sample(n, random_state=args.seed))
        df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df = make_splits(df, args.seed)
    split_path = output_dir / "v11_splits.csv"
    df[["source_row_id", "human_label", "target_m", "group_key", "split"]].to_csv(split_path, index=False, encoding="utf-8-sig")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    holdout_df = df[df["split"] == "holdout"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "Organic", 1: "Manipulative"},
        label2id={"Organic": 0, "Manipulative": 1},
    ).to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["target_m"].to_numpy())
    val_ds = TextDataset(val_df["text"].tolist(), val_df["target_m"].to_numpy())
    holdout_ds = TextDataset(holdout_df["text"].tolist(), holdout_df["target_m"].to_numpy())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    counts = train_df["target_m"].value_counts().to_dict()
    total = len(train_df)
    class_weights = torch.tensor(
        [total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_pr_auc = -1.0
    best_epoch = -1
    patience_left = args.early_stopping_patience
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=running_loss / step)

        val_prob = predict_probabilities(model, val_loader, device, use_amp)
        threshold_info = choose_recall_first_threshold(val_df["target_m"].to_numpy(), val_prob, args.min_recall)
        val_metrics = metric_block(val_df["target_m"].to_numpy(), val_prob, threshold_info["threshold"])
        epoch_payload = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_loader)), "validation": val_metrics}
        history.append(epoch_payload)
        common.write_json(output_dir / "training_history.json", {"history": history})
        current_pr_auc = float(val_metrics["pr_auc"] or 0.0)
        if current_pr_auc > best_pr_auc:
            best_pr_auc = current_pr_auc
            best_epoch = epoch
            patience_left = args.early_stopping_patience
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    val_prob = predict_probabilities(model, val_loader, device, use_amp)
    threshold_info = choose_recall_first_threshold(val_df["target_m"].to_numpy(), val_prob, args.min_recall)
    threshold = threshold_info["threshold"]
    holdout_prob = predict_probabilities(model, holdout_loader, device, use_amp)
    bert_metrics = {
        "best_epoch": best_epoch,
        "validation": metric_block(val_df["target_m"].to_numpy(), val_prob, threshold),
        "validation_review": review_block(val_df["target_m"].to_numpy(), val_prob, threshold, args.review_margin),
        "holdout": metric_block(holdout_df["target_m"].to_numpy(), holdout_prob, threshold),
        "holdout_review": review_block(holdout_df["target_m"].to_numpy(), holdout_prob, threshold, args.review_margin),
    }

    tfidf_payload: dict[str, Any] | None = None
    final_strategy = "bert"
    ensemble_weight_bert = 1.0
    if not args.skip_tfidf:
        tfidf_payload = train_tfidf(train_df, val_df, holdout_df, output_dir)
        tfidf_val = tfidf_payload["validation_prob"]
        best_ensemble: dict[str, Any] | None = None
        for weight in np.linspace(0.50, 0.95, 10):
            mixed = weight * val_prob + (1.0 - weight) * tfidf_val
            info = choose_recall_first_threshold(val_df["target_m"].to_numpy(), mixed, args.min_recall)
            metrics = metric_block(val_df["target_m"].to_numpy(), mixed, info["threshold"])
            candidate = {"weight_bert": float(weight), "threshold": info["threshold"], "metrics": metrics}
            if best_ensemble is None or float(metrics["pr_auc"] or 0.0) > float(best_ensemble["metrics"]["pr_auc"] or 0.0):
                best_ensemble = candidate
        if best_ensemble:
            bert_pr = float(bert_metrics["validation"]["pr_auc"] or 0.0)
            bert_f1 = float(bert_metrics["validation"]["f1_M"])
            ens_pr = float(best_ensemble["metrics"]["pr_auc"] or 0.0)
            ens_f1 = float(best_ensemble["metrics"]["f1_M"])
            if ens_pr >= bert_pr + 0.01 or ens_f1 >= bert_f1 + 0.01:
                final_strategy = "ensemble"
                ensemble_weight_bert = float(best_ensemble["weight_bert"])
                threshold = float(best_ensemble["threshold"])

    config = {
        "version": "v11_text_first_bert",
        "model_name": args.model_name,
        "max_length": args.max_length,
        "label_mapping": {"O": 0, "M": 1},
        "threshold": float(threshold),
        "review_margin": float(args.review_margin),
        "min_recall_target": float(args.min_recall),
        "final_strategy": final_strategy,
        "ensemble_weight_bert": float(ensemble_weight_bert),
        "uses_metadata_for_prediction": False,
        "text_feature": "original_text",
        "model_dir": str(model_dir),
        "tfidf_bundle": str(output_dir / "tfidf_baseline.joblib"),
        "nearest_index": str(output_dir / "nearest_examples.joblib"),
    }
    common.write_json(model_dir / "v11_config.json", config)
    metrics = {
        "dataset": {
            "rows_after_blank_drop": int(len(df)),
            "splits": {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()},
            "labels": {str(k): int(v) for k, v in df["human_label"].value_counts().to_dict().items()},
        },
        "cuda": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        },
        "bert": bert_metrics,
        "tfidf": tfidf_payload["metrics"] if tfidf_payload else None,
        "final_config": config,
    }
    common.write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=common.json_default))
    return metrics


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        sys.exit(130)
