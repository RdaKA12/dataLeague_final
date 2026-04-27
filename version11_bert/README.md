# Version 11 Text-First BERT

V11 is a supervised, text-only MO classifier trained on the manually labeled
60k dataset. It is designed for jury live inference where only unseen text may
be provided.

## Core idea

- Final decision model uses `original_text` only.
- Metadata (`language`, `url`, `author_hash`, `primary_theme`, `date`) is used
  for dashboard and full-dataset analysis, not as live prediction features.
- Labels come only from:
  `version2_experiments/labeling/datathon_manual_labeled_mo_60k.csv`
- Label mapping:
  - `O -> Organic`
  - `M -> Manipulative`

## GPU setup

The machine has an NVIDIA GPU, but the global Python may have CPU-only PyTorch.
Use the CUDA wheel before training:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r version11_bert\requirements.txt
python version11_bert\scripts\v11_cuda_check.py
```

Training refuses to run without CUDA unless `--allow-cpu` is passed for a tiny
smoke test.

## Train

```powershell
python version11_bert\scripts\v11_train.py
```

Useful faster smoke test:

```powershell
python version11_bert\scripts\v11_train.py --sample-size 512 --epochs 1 --max-length 64 --allow-cpu --skip-tfidf --output-dir version11_bert\processed\smoke
```

## CLI inference

```powershell
python version11_bert\scripts\v11_inference.py --text "Free crypto giveaway, join our Telegram now." --json
```

## Full 5M scoring

```powershell
python version11_bert\scripts\v11_score_full.py --batch-size 512
python version11_bert\scripts\v11_score_full.py --combine
```

Scoring writes resumable row-group shards under:

```text
version11_bert/processed/full_score_shards/
```

Combined output:

```text
version11_bert/processed/v11_full_scored.parquet
```

## Analytics and jury report

After scoring is complete, build the dashboard-ready aggregate CSV files and
the concise jury delivery report:

```powershell
python version11_bert\scripts\v11_build_analytics.py
```

Outputs:

```text
version11_bert/outputs/analytics/
version11_bert/V11_JURY_DELIVERY_REPORT.md
```

## Streamlit dashboard and live demo

```powershell
streamlit run version11_bert\scripts\v11_dashboard.py
```

The app supports:

- Single-text live prediction
- Batch CSV prediction with input columns `test_id,text`
- Platform x language risk map
- Theme risk distribution
- Time/burst trend
- Author risk table
- High-risk examples

Batch CSV input example:

```csv
test_id,text
TEST_0000,"I got a likely phishing message about a fake crypto giveaway."
TEST_0001,"No transactions on weekends; I will check it again tomorrow."
```

Sample file:

```text
version11_bert/examples/v11_batch_test_sample.csv
```

Batch outputs:

- `v11_batch_predictions_detailed.csv`: original columns plus scores, decisions, confidence, and review flag.
- `v11_batch_submission.csv`: `test_id,label,manipulative_score,organic_score,review_flag`.
