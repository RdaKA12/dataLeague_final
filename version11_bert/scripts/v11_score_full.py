"""Score datathonFINAL.parquet with the V11 model in resumable row-group shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import v11_common as common
from v11_inference import load_bundle, final_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the full datathon parquet with V11")
    parser.add_argument("--input", type=Path, default=common.DEFAULT_FULL_DATA)
    parser.add_argument("--model-dir", type=Path, default=common.DEFAULT_MODEL_DIR)
    parser.add_argument("--shard-dir", type=Path, default=common.DEFAULT_PROCESSED / "full_score_shards")
    parser.add_argument("--output", type=Path, default=common.DEFAULT_PROCESSED / "v11_full_scored.parquet")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-row-group", type=int, default=0)
    parser.add_argument("--max-row-groups", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--combine", action="store_true")
    return parser.parse_args()


def combine_shards(shard_dir: Path, output: Path) -> dict[str, object]:
    shards = sorted(shard_dir.glob("part_rg_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No shards found in {shard_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for shard in tqdm(shards, desc="combine"):
            table = pq.read_table(shard)
            rows += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return {"output": str(output), "shards": len(shards), "rows": rows}


def score_full() -> dict[str, object]:
    args = parse_args()
    if args.combine:
        payload = combine_shards(args.shard_dir, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    bundle = load_bundle(str(args.model_dir.resolve()))
    config = bundle["config"]
    threshold = float(config["threshold"])
    review_margin = float(config.get("review_margin", 0.08))
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.input)
    row_offsets = [0]
    for rg in range(pf.num_row_groups):
        row_offsets.append(row_offsets[-1] + pf.metadata.row_group(rg).num_rows)
    end_rg = pf.num_row_groups
    if args.max_row_groups:
        end_rg = min(end_rg, args.start_row_group + args.max_row_groups)

    written = 0
    skipped = 0
    for rg in tqdm(range(args.start_row_group, end_rg), desc="row-groups"):
        out_path = args.shard_dir / f"part_rg_{rg:04d}.parquet"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        texts = df["original_text"].fillna("").astype(str).map(common.compact_text).tolist()
        probs: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), args.batch_size), desc=f"rg {rg}", leave=False):
            probs.append(final_probabilities(texts[start : start + args.batch_size], bundle, batch_size=args.batch_size))
        probability = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
        decision = np.where(probability >= threshold, "Manipulative", "Organic")
        review = np.abs(probability - threshold) < review_margin
        out = pd.DataFrame(
            {
                "row_id": np.arange(row_offsets[rg], row_offsets[rg + 1], dtype=np.int64),
                "original_text": df.get("original_text", ""),
                "language": df.get("language", ""),
                "url": df.get("url", ""),
                "author_hash": df.get("author_hash", ""),
                "date": df.get("date", ""),
                "primary_theme": df.get("primary_theme", ""),
                "manipulative_score": probability.astype(np.float32),
                "organic_score": (1.0 - probability).astype(np.float32),
                "decision": decision,
                "review_flag": review,
            }
        )
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), out_path, compression="zstd")
        written += 1
    payload = {
        "input": str(args.input),
        "shard_dir": str(args.shard_dir),
        "row_groups_total": pf.num_row_groups,
        "row_groups_written": written,
        "row_groups_skipped": skipped,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    score_full()

