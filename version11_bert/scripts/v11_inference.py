"""CLI and importable live inference for V11."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import v11_common as common
from v11_text_signals import compact_text, extract_signals


@lru_cache(maxsize=2)
def load_bundle(model_dir_text: str = str(common.DEFAULT_MODEL_DIR.resolve())) -> dict[str, Any]:
    model_dir = Path(model_dir_text)
    config = common.read_json(model_dir / "v11_config.json")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tfidf_bundle = None
    tfidf_path = Path(config.get("tfidf_bundle", ""))
    if config.get("final_strategy") == "ensemble" and tfidf_path.exists():
        tfidf_bundle = joblib.load(tfidf_path)
    nearest_index = None
    nearest_path = Path(config.get("nearest_index", ""))
    if nearest_path.exists():
        nearest_index = joblib.load(nearest_path)
    return {
        "config": config,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "tfidf_bundle": tfidf_bundle,
        "nearest_index": nearest_index,
    }


def bert_probabilities(texts: list[str], bundle: dict[str, Any], batch_size: int = 32) -> np.ndarray:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    max_length = int(bundle["config"]["max_length"])
    probs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(**enc).logits
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)


def final_probabilities(texts: list[str], bundle: dict[str, Any], batch_size: int = 32) -> np.ndarray:
    clean = [compact_text(t) for t in texts]
    bert_probs = bert_probabilities(clean, bundle, batch_size=batch_size)
    config = bundle["config"]
    if config.get("final_strategy") != "ensemble" or bundle.get("tfidf_bundle") is None:
        return bert_probs
    tfidf = bundle["tfidf_bundle"]
    tfidf_probs = tfidf["classifier"].predict_proba(tfidf["vectorizer"].transform(clean))[:, 1]
    weight = float(config.get("ensemble_weight_bert", 1.0))
    return weight * bert_probs + (1.0 - weight) * tfidf_probs


def nearest_examples(text: str, bundle: dict[str, Any], top_n: int = 3) -> dict[str, list[dict[str, Any]]]:
    index = bundle.get("nearest_index")
    if index is None:
        return {"nearest_manipulative": [], "nearest_organic": []}
    vector = index["vectorizer"].transform([compact_text(text)])
    distances, indices = index["neighbors"].kneighbors(vector, n_neighbors=min(40, len(index["examples"])))
    out = {"nearest_manipulative": [], "nearest_organic": []}
    for distance, idx in zip(distances[0], indices[0]):
        item = index["examples"][int(idx)]
        row = {
            "similarity": round(float(1.0 - distance), 4),
            "label": item.get("human_label", ""),
            "source_row_id": item.get("source_row_id", ""),
            "note": item.get("labeler_notes", ""),
            "language": item.get("language", ""),
            "theme": item.get("primary_theme", ""),
            "platform": item.get("url", ""),
            "text_excerpt": compact_text(item.get("text", ""))[:220],
        }
        if int(item.get("target_m", 0)) == 1 and len(out["nearest_manipulative"]) < top_n:
            out["nearest_manipulative"].append(row)
        elif int(item.get("target_m", 0)) == 0 and len(out["nearest_organic"]) < top_n:
            out["nearest_organic"].append(row)
        if len(out["nearest_manipulative"]) >= top_n and len(out["nearest_organic"]) >= top_n:
            break
    return out


def predict_text(text: str, model_dir: Path = common.DEFAULT_MODEL_DIR) -> dict[str, Any]:
    bundle = load_bundle(str(model_dir.resolve()))
    config = bundle["config"]
    clean = compact_text(text)
    signals = extract_signals(clean)
    if not clean:
        return {
            "decision": "Organic",
            "decision_tr": "Organik",
            "manipulative_score": 0.0,
            "organic_score": 1.0,
            "confidence": 0.0,
            "review_flag": True,
            "threshold": config.get("threshold"),
            "summary_tr": "Metin boş olduğu için güven düşük; manuel review önerilir.",
            "reasons": ["Boş metin; model kararı güvenilir değildir"],
            "signals": signals,
            "nearest_manual_examples": {"nearest_manipulative": [], "nearest_organic": []},
            "model_version": config.get("version", "v11"),
        }
    probability = float(final_probabilities([clean], bundle, batch_size=1)[0])
    threshold = float(config["threshold"])
    review_margin = float(config.get("review_margin", 0.08))
    decision = common.decision_from_probability(probability, threshold, review_margin)
    reasons = []
    if decision["review_flag"]:
        reasons.append("Skor karar eşiğine yakın; manuel review önerilir")
    elif decision["decision"] == "Manipulative":
        reasons.append("V11 text-only BERT metni manipülatif örüntülere daha yakın buldu")
    else:
        reasons.append("V11 text-only BERT metni organik örüntülere daha yakın buldu")
    reasons.extend(signals["reasons"])
    reasons.append(f"manipulative_score={probability:.3f}, threshold={threshold:.3f}")
    return {
        **decision,
        "manipulative_score": round(probability, 4),
        "organic_score": round(1.0 - probability, 4),
        "threshold": round(threshold, 4),
        "review_margin": round(review_margin, 4),
        "summary_tr": (
            f"Karar: {decision['decision_tr']}. Manipülatif skor {probability:.3f}, "
            f"organiklik {1.0 - probability:.3f}."
        ),
        "reasons": list(dict.fromkeys(reasons)),
        "signals": signals,
        "nearest_manual_examples": nearest_examples(clean, bundle),
        "model_version": config.get("version", "v11"),
        "final_strategy": config.get("final_strategy", "bert"),
    }


def predict_many(texts: list[str], model_dir: Path = common.DEFAULT_MODEL_DIR, batch_size: int = 32) -> list[dict[str, Any]]:
    bundle = load_bundle(str(model_dir.resolve()))
    config = bundle["config"]
    threshold = float(config["threshold"])
    review_margin = float(config.get("review_margin", 0.08))
    clean = [compact_text(t) for t in texts]
    probabilities = final_probabilities(clean, bundle, batch_size=batch_size)
    results = []
    for text, probability in zip(clean, probabilities):
        decision = common.decision_from_probability(float(probability), threshold, review_margin)
        results.append(
            {
                **decision,
                "manipulative_score": float(probability),
                "organic_score": float(1.0 - probability),
                "threshold": threshold,
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V11 live inference")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path, default=common.DEFAULT_MODEL_DIR)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    result = predict_text(args.text, args.model_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=common.json_default))
    else:
        print(result["summary_tr"])
        print("reasons=" + "; ".join(result["reasons"]))


if __name__ == "__main__":
    main()

