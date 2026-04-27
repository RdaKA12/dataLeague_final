# Model and Large Artefacts

The GitHub repository intentionally excludes large binary artefacts and the
challenge dataset.

Add the final Drive link here before submission:

```text
MODEL_DRIVE_LINK=https://drive.google.com/drive/folders/1RDjpVpzjCuMjm3iuUnyWWGspU1RrHlds?usp=sharing
```

Drive folder:
https://drive.google.com/drive/folders/1RDjpVpzjCuMjm3iuUnyWWGspU1RrHlds?usp=sharing

Expected artefact placement after download:

```text
version11_bert/processed/model/config.json
version11_bert/processed/model/model.safetensors
version11_bert/processed/model/special_tokens_map.json
version11_bert/processed/model/tokenizer.json
version11_bert/processed/model/tokenizer_config.json
version11_bert/processed/model/v11_config.json
version11_bert/processed/nearest_examples.joblib
version11_bert/processed/tfidf_baseline.joblib
version11_bert/processed/v11_full_scored.parquet
```

Minimum required for live inference:

```text
version11_bert/processed/model/
```

Optional but useful:

- `nearest_examples.joblib`: enables nearest manual examples in explanations.
- `v11_full_scored.parquet`: enables full dashboard drill-down locally.
- `tfidf_baseline.joblib`: keeps the fast baseline available for comparison.

The dashboard can still show aggregate analytics from committed CSV/JSON files
without `v11_full_scored.parquet`.
