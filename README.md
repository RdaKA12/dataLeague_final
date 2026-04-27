# DataLeague Final - V11 Manipulation Detection

This repository contains the final V11 code delivery for the DataLeague social
media manipulation and anomaly detection task.

## Delivered outputs

- Reliability / organicity scoring algorithm: text-only `xlm-roberta-base`
  binary classifier that returns `manipulative_score`, `organic_score`,
  decision, confidence, and review flag.
- Manipulation map dashboard: platform, language, theme, time, author, and
  duplicate-cluster risk views built from the scored 5M dataset.
- Live inference pipeline: CLI and Streamlit interface for unseen jury texts.

## Main commands

Install dependencies:

```powershell
python -m pip install -r version11_bert\requirements.txt
```

Check CUDA:

```powershell
python version11_bert\scripts\v11_cuda_check.py
```

Run live CLI inference after downloading the model artefacts:

```powershell
python version11_bert\scripts\v11_inference.py --text "Free crypto giveaway, join now." --json
```

Run the Streamlit demo/dashboard:

```powershell
python -m streamlit run version11_bert\scripts\v11_dashboard.py
```

Score the full parquet dataset locally:

```powershell
python version11_bert\scripts\v11_score_full.py --batch-size 512
python version11_bert\scripts\v11_score_full.py --combine
python version11_bert\scripts\v11_build_analytics.py
```

## Model artefacts

Large artefacts are not committed to GitHub. Download them from the Drive link
provided in `MODEL_ARTIFACTS.md` and place them at the paths described there.

The most important local artefact paths are:

- `version11_bert/processed/model/`
- `version11_bert/processed/nearest_examples.joblib`
- `version11_bert/processed/v11_full_scored.parquet`

## Current model summary

- Training labels: 59,867 non-empty manually labelled rows.
- Model: `xlm-roberta-base`, text-only, `original_text`.
- GPU used: NVIDIA RTX 4070 Laptop GPU, CUDA PyTorch.
- Holdout ROC-AUC: `0.9248`
- Holdout PR-AUC: `0.7059`
- External validation ROC-AUC: `0.9430`
- External validation PR-AUC: `0.8919`
- External validation M recall: `0.8992`

Full delivery details are in:

- `version11_bert/V11_JURY_DELIVERY_REPORT.md`
- `version11_bert/outputs/analytics/analytics_summary.json`
