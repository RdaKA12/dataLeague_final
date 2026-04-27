"""Shared utilities for V11 BERT training, inference, and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[0]

DEFAULT_LABELS = PROJECT_ROOT / "version2_experiments" / "labeling" / "datathon_manual_labeled_mo_60k.csv"
DEFAULT_FULL_DATA = PROJECT_ROOT / "veriseti" / "datathonFINAL.parquet"
DEFAULT_PROCESSED = ROOT / "processed"
DEFAULT_MODEL_DIR = DEFAULT_PROCESSED / "model"
DEFAULT_TFIDF_BUNDLE = DEFAULT_PROCESSED / "tfidf_baseline.joblib"
DEFAULT_NEAREST_INDEX = DEFAULT_PROCESSED / "nearest_examples.joblib"
DEFAULT_METRICS = DEFAULT_PROCESSED / "metrics.json"
DEFAULT_CONFIG = DEFAULT_MODEL_DIR / "v11_config.json"

LABEL_TO_ID = {"O": 0, "M": 1}
ID_TO_LABEL = {0: "O", 1: "M"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def compact_text(value: str | None) -> str:
    return " ".join(str(value or "").split())


def normalized_text_hash(text: str) -> str:
    return hashlib.sha1(compact_text(text).lower().encode("utf-8", errors="ignore")).hexdigest()


def connected_group_keys(authors: pd.Series, text_hashes: pd.Series) -> list[str]:
    """Build leakage-safe components over author and exact normalized text.

    A row is connected to its author node and exact-text node. This keeps all
    posts from the same author in one split and also keeps exact duplicate text
    across authors in one split.
    """

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    row_nodes: list[str] = []
    for idx, (author, text_hash) in enumerate(zip(authors.fillna("").astype(str), text_hashes.fillna("").astype(str))):
        row_node = f"row:{idx}"
        author_node = f"author:{author}" if author else row_node
        text_node = f"text:{text_hash}" if text_hash else row_node
        union(row_node, author_node)
        union(row_node, text_node)
        row_nodes.append(row_node)
    return [find(node) for node in row_nodes]


def load_labels(path: Path = DEFAULT_LABELS) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"original_text", "human_label", "author_hash"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required label columns: {missing}")
    df = df[df["human_label"].isin(["O", "M"])].copy()
    df["text"] = df["original_text"].map(compact_text)
    df = df[df["text"].str.len() > 0].copy()
    df["target_m"] = df["human_label"].map(LABEL_TO_ID).astype(int)
    df["text_hash"] = df["text"].map(normalized_text_hash)
    df["group_key"] = connected_group_keys(df["author_hash"], df["text_hash"])
    return df.reset_index(drop=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def decision_from_probability(probability: float, threshold: float, review_margin: float) -> dict[str, Any]:
    review_flag = abs(probability - threshold) < review_margin
    decision = "Manipulative" if probability >= threshold else "Organic"
    decision_tr = "Manipulatif" if decision == "Manipulative" else "Organik"
    denominator = max(threshold, 1.0 - threshold, 1e-6)
    confidence = min(1.0, abs(probability - threshold) / denominator)
    return {
        "decision": decision,
        "decision_tr": decision_tr,
        "review_flag": bool(review_flag),
        "confidence": float(confidence),
    }
