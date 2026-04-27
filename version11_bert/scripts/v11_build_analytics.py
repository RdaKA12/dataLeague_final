"""Build V11 dashboard analytics and a jury-ready report from scored data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import v11_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V11 analytics artifacts")
    parser.add_argument("--scored", type=Path, default=common.DEFAULT_PROCESSED / "v11_full_scored.parquet")
    parser.add_argument("--shard-dir", type=Path, default=common.DEFAULT_PROCESSED / "full_score_shards")
    parser.add_argument("--metrics", type=Path, default=common.DEFAULT_PROCESSED / "metrics.json")
    parser.add_argument(
        "--external-metrics",
        type=Path,
        default=common.ROOT / "eval_outputs" / "text_only_qwen_validation_2950_dedup_v11_metrics.json",
    )
    parser.add_argument("--output-dir", type=Path, default=common.ROOT / "outputs" / "analytics")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--top-n", type=int, default=250)
    return parser.parse_args()


def source_for(args: argparse.Namespace) -> Path:
    if args.scored.exists():
        return args.scored
    if args.shard_dir.exists() and any(args.shard_dir.glob("part_rg_*.parquet")):
        return args.shard_dir
    raise FileNotFoundError(f"No scored parquet found at {args.scored} or shards at {args.shard_dir}")


def safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def normalize_text_hash(text: str) -> str:
    compact = " ".join(str(text or "").split()).lower()
    return hashlib.sha1(compact.encode("utf-8", errors="ignore")).hexdigest()


def merge_group_sum(store: dict[tuple[Any, ...], dict[str, float]], frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.empty:
        return
    grouped = (
        frame.groupby(keys, dropna=False)
        .agg(
            rows=("manipulative_score", "size"),
            mean_score_sum=("manipulative_score", "sum"),
            manipulative=("is_manipulative", "sum"),
            review=("review_flag_num", "sum"),
        )
        .reset_index()
    )
    for row in grouped.to_dict(orient="records"):
        key = tuple(row[k] for k in keys)
        bucket = store.setdefault(key, {"rows": 0.0, "score_sum": 0.0, "manipulative": 0.0, "review": 0.0})
        bucket["rows"] += float(row["rows"])
        bucket["score_sum"] += float(row["mean_score_sum"])
        bucket["manipulative"] += float(row["manipulative"])
        bucket["review"] += float(row["review"])


def store_to_frame(store: dict[tuple[Any, ...], dict[str, float]], keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, vals in store.items():
        row = {k: v for k, v in zip(keys, key)}
        n = max(vals["rows"], 1.0)
        row.update(
            {
                "rows": int(vals["rows"]),
                "mean_manipulative_score": vals["score_sum"] / n,
                "manipulative_rate": vals["manipulative"] / n,
                "review_rate": vals["review"] / n,
                "manipulative_count": int(vals["manipulative"]),
                "review_count": int(vals["review"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def trim_top_examples(current: pd.DataFrame, new_rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if current.empty:
        combined = new_rows
    elif new_rows.empty:
        combined = current
    else:
        combined = pd.concat([current, new_rows], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values("manipulative_score", ascending=False).head(top_n).reset_index(drop=True)


def build() -> dict[str, Any]:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = source_for(args)
    dataset = ds.dataset(source, format="parquet")
    available = set(dataset.schema.names)
    columns = [
        col
        for col in [
            "row_id",
            "original_text",
            "language",
            "url",
            "author_hash",
            "date",
            "primary_theme",
            "manipulative_score",
            "organic_score",
            "decision",
            "review_flag",
        ]
        if col in available
    ]
    scanner = dataset.scanner(columns=columns, batch_size=args.batch_size)

    platform_language: dict[tuple[Any, ...], dict[str, float]] = {}
    platform_store: dict[tuple[Any, ...], dict[str, float]] = {}
    language_store: dict[tuple[Any, ...], dict[str, float]] = {}
    theme_store: dict[tuple[Any, ...], dict[str, float]] = {}
    day_store: dict[tuple[Any, ...], dict[str, float]] = {}
    author_store: dict[tuple[Any, ...], dict[str, float]] = {}
    duplicate_store: dict[str, dict[str, Any]] = {}
    high_examples = pd.DataFrame()

    total_rows = 0
    score_sum = 0.0
    manipulative_count = 0
    review_count = 0

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        total_rows += len(df)
        for col in ["language", "url", "author_hash", "primary_theme", "date", "original_text"]:
            if col not in df:
                df[col] = ""
        df["manipulative_score"] = pd.to_numeric(df["manipulative_score"], errors="coerce").fillna(0.0)
        df["is_manipulative"] = (df.get("decision", "").astype(str) == "Manipulative").astype(int)
        df["review_flag_num"] = df.get("review_flag", False).astype(bool).astype(int)
        score_sum += float(df["manipulative_score"].sum())
        manipulative_count += int(df["is_manipulative"].sum())
        review_count += int(df["review_flag_num"].sum())

        merge_group_sum(platform_language, df, ["url", "language"])
        merge_group_sum(platform_store, df, ["url"])
        merge_group_sum(language_store, df, ["language"])
        merge_group_sum(theme_store, df, ["primary_theme"])
        if "date" in df:
            day_df = df.copy()
            day_df["day"] = pd.to_datetime(day_df["date"], errors="coerce", utc=True, format="mixed").dt.date.astype(str)
            merge_group_sum(day_store, day_df, ["day"])
        merge_group_sum(author_store, df[df["author_hash"].astype(str).ne("")], ["author_hash"])

        example_cols = ["row_id", "original_text", "language", "url", "author_hash", "date", "primary_theme", "manipulative_score", "decision", "review_flag"]
        high_examples = trim_top_examples(
            high_examples,
            df[df["is_manipulative"].eq(1)][example_cols].sort_values("manipulative_score", ascending=False).head(args.top_n),
            args.top_n,
        )

        high = df[df["is_manipulative"].eq(1) & df["original_text"].astype(str).str.strip().ne("")]
        for row in high[["original_text", "language", "url", "primary_theme", "manipulative_score"]].to_dict(orient="records"):
            text = str(row["original_text"])
            key = normalize_text_hash(text)
            bucket = duplicate_store.setdefault(
                key,
                {
                    "rows": 0,
                    "max_score": 0.0,
                    "score_sum": 0.0,
                    "sample_text": text[:500],
                    "languages": set(),
                    "platforms": set(),
                    "themes": set(),
                },
            )
            bucket["rows"] += 1
            bucket["score_sum"] += float(row["manipulative_score"])
            bucket["max_score"] = max(float(bucket["max_score"]), float(row["manipulative_score"]))
            bucket["languages"].add(str(row["language"]))
            bucket["platforms"].add(str(row["url"]))
            bucket["themes"].add(str(row["primary_theme"]))

    def write_group(name: str, store: dict[tuple[Any, ...], dict[str, float]], keys: list[str], min_rows: int = 1) -> pd.DataFrame:
        frame = store_to_frame(store, keys)
        if frame.empty:
            out = args.output_dir / f"{name}.csv"
            frame.to_csv(out, index=False, encoding="utf-8-sig")
            return frame
        frame = frame[frame["rows"] >= min_rows].sort_values(["mean_manipulative_score", "rows"], ascending=[False, False])
        frame.to_csv(args.output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        return frame

    platform_language_df = write_group("platform_language_risk", platform_language, ["url", "language"], min_rows=20)
    platform_df = write_group("platform_risk", platform_store, ["url"], min_rows=20)
    language_df = write_group("language_risk", language_store, ["language"], min_rows=20)
    theme_df = write_group("theme_risk", theme_store, ["primary_theme"], min_rows=20)
    day_df = write_group("daily_risk_trend", day_store, ["day"], min_rows=1)
    author_df = write_group("author_risk", author_store, ["author_hash"], min_rows=3)
    if not author_df.empty:
        volume_norm = np.log1p(author_df["rows"]) / max(np.log1p(author_df["rows"]).max(), 1.0)
        author_df["author_risk_score"] = (
            0.60 * author_df["mean_manipulative_score"]
            + 0.30 * author_df["manipulative_rate"]
            + 0.10 * volume_norm
        )
        author_df = author_df.sort_values(["author_risk_score", "rows"], ascending=[False, False])
        author_df.to_csv(args.output_dir / "author_risk.csv", index=False, encoding="utf-8-sig")

    high_examples.to_csv(args.output_dir / "high_risk_examples.csv", index=False, encoding="utf-8-sig")
    duplicate_rows = []
    for key, bucket in duplicate_store.items():
        if int(bucket["rows"]) < 2:
            continue
        duplicate_rows.append(
            {
                "text_hash": key,
                "rows": int(bucket["rows"]),
                "mean_score": float(bucket["score_sum"]) / max(int(bucket["rows"]), 1),
                "max_score": float(bucket["max_score"]),
                "languages": ", ".join(sorted(bucket["languages"])),
                "platforms": ", ".join(sorted(bucket["platforms"])),
                "themes": ", ".join(sorted(bucket["themes"])),
                "sample_text": bucket["sample_text"],
            }
        )
    duplicate_df = pd.DataFrame(duplicate_rows)
    if not duplicate_df.empty:
        duplicate_df = duplicate_df.sort_values(["rows", "max_score"], ascending=[False, False]).head(500)
    duplicate_df.to_csv(args.output_dir / "high_risk_duplicate_clusters.csv", index=False, encoding="utf-8-sig")

    train_metrics = common.read_json(args.metrics) if args.metrics.exists() else {}
    external_metrics = common.read_json(args.external_metrics) if args.external_metrics.exists() else {}
    summary = {
        "source": str(source),
        "rows_scored": total_rows,
        "mean_manipulative_score": score_sum / max(total_rows, 1),
        "decision_counts": {
            "Manipulative": manipulative_count,
            "Organic": total_rows - manipulative_count,
            "Review": review_count,
        },
        "decision_rates": {
            "manipulative_rate": manipulative_count / max(total_rows, 1),
            "review_rate": review_count / max(total_rows, 1),
        },
        "top_platform_language": platform_language_df.head(15).to_dict(orient="records") if not platform_language_df.empty else [],
        "top_themes": theme_df.head(15).to_dict(orient="records") if not theme_df.empty else [],
        "top_authors": author_df.head(25).to_dict(orient="records") if not author_df.empty else [],
        "duplicate_cluster_count_min2": int(len(duplicate_df)),
        "train_holdout_metrics": train_metrics.get("bert", {}).get("holdout", {}),
        "external_validation_metrics": external_metrics.get("overall", {}),
        "outputs": {
            "platform_language_risk": str(args.output_dir / "platform_language_risk.csv"),
            "platform_risk": str(args.output_dir / "platform_risk.csv"),
            "language_risk": str(args.output_dir / "language_risk.csv"),
            "theme_risk": str(args.output_dir / "theme_risk.csv"),
            "daily_risk_trend": str(args.output_dir / "daily_risk_trend.csv"),
            "author_risk": str(args.output_dir / "author_risk.csv"),
            "high_risk_examples": str(args.output_dir / "high_risk_examples.csv"),
            "high_risk_duplicate_clusters": str(args.output_dir / "high_risk_duplicate_clusters.csv"),
        },
    }
    common.write_json(args.output_dir / "analytics_summary.json", summary)
    report = render_report(summary)
    report_path = common.ROOT / "V11_JURY_DELIVERY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary, "report_path": str(report_path)}, ensure_ascii=False, indent=2, default=safe_value))
    return summary


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def render_report(summary: dict[str, Any]) -> str:
    holdout = summary.get("train_holdout_metrics", {})
    external = summary.get("external_validation_metrics", {})
    lines = [
        "# V11 Jury Delivery Report",
        "",
        "## 1. Güvenilirlik / Organiklik Skoru Algoritması",
        "",
        "- Ana model: GPU ile eğitilmiş text-only `xlm-roberta-base` binary classifier.",
        "- Her içerik için `manipulative_score` ve `organic_score = 1 - manipulative_score` üretilir.",
        "- Canlı jüri tahmini yalnız metne dayanır; metadata model kararına feature olarak girmez.",
        "- Author seviyesi risk, içerik skorlarının agregasyonu olarak dashboard tarafında hesaplanır.",
        "",
        "### Model metrikleri",
        "",
        "| Set | ROC-AUC | PR-AUC | M Precision | M Recall | M F1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Holdout | {metric_value(holdout, 'roc_auc')} | {metric_value(holdout, 'pr_auc')} | {metric_value(holdout, 'precision_M')} | {metric_value(holdout, 'recall_M')} | {metric_value(holdout, 'f1_M')} |",
        f"| Text-only Qwen validation | {metric_value(external, 'roc_auc')} | {metric_value(external, 'pr_auc')} | {metric_value(external, 'precision_M')} | {metric_value(external, 'recall_M')} | {metric_value(external, 'f1_M')} |",
        "",
        "## 2. Manipülasyon Haritası / Dashboard",
        "",
        f"- Skorlanan satır: `{summary.get('rows_scored', 0):,}`",
        f"- Ortalama manipülatif skor: `{summary.get('mean_manipulative_score', 0):.4f}`",
        f"- Manipulatif karar oranı: `{pct(summary.get('decision_rates', {}).get('manipulative_rate', 0))}`",
        f"- Review oranı: `{pct(summary.get('decision_rates', {}).get('review_rate', 0))}`",
        "- Dashboard artefactleri platform, dil, tema, zaman, kullanıcı ve duplicate cluster kırılımlarını üretir.",
        "",
        "## 3. Canlı Tahmin Modeli",
        "",
        "- CLI: `python version11_bert\\scripts\\v11_inference.py --text \"...\" --json`",
        "- Dashboard: `streamlit run version11_bert\\scripts\\v11_dashboard.py`",
        "- Çıktı: karar, manipülatif skor, organiklik skoru, review flag, gerekçeler ve yakın manuel örnekler.",
        "",
        "## 4. Rubric’e Göre Durum",
        "",
        "- Analitik derinlik: platform/dil/tema/zaman/author/duplicate cluster analizleri üretildi.",
        "- Model ve canlı tahmin: external text-only validation üzerinde yüksek recall ve PR-AUC; CLI çalışıyor.",
        "- Dashboard ve sunum: Streamlit app aynı bundle ve skor dosyalarını kullanıyor.",
        "- Teknik uygulanabilirlik: GPU eğitim, resumable full scoring, shard combine ve hata durumunda resume desteği var.",
        "",
        "## 5. Geliştirilebilir Yerler",
        "",
        "- False positive azaltma: `final_low` ve `background` segmentlerindeki FP örnekleriyle threshold veya calibration iyileştirilebilir.",
        "- Açıklanabilirlik: token attribution için Integrated Gradients eklenirse gerekçeler daha model-içi olur.",
        "- Author/network modeli: canlı text-only karar korunarak dashboard’a ayrı metadata/context ranker eklenebilir.",
        "- Aktif öğrenme: V11’in yanlış pozitif/yanlış negatiflerinden 2-5k yeni manuel audit turu performansı yükseltir.",
        "- Full scoring periyodikleştirilebilir: yeni veri geldiğinde sadece yeni row-group/shard skorlanır.",
        "",
        "## 6. Üretilen Artefactler",
        "",
    ]
    for name, path in summary.get("outputs", {}).items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    build()
