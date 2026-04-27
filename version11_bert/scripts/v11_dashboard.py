"""Streamlit dashboard and live demo for V11."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

import v11_common as common
from v11_inference import predict_many, predict_text


BATCH_TEXT_COLUMNS = ["text", "original_text", "content", "body"]
BATCH_ID_COLUMNS = ["test_id", "id", "row_id"]


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def build_batch_output(df: pd.DataFrame, results: list[dict]) -> pd.DataFrame:
    result_df = pd.DataFrame(results)
    labels = result_df["manipulative_score"].astype(float).clip(0.0, 1.0)
    id_col = resolve_column(df, BATCH_ID_COLUMNS)
    text_col = resolve_column(df, BATCH_TEXT_COLUMNS)

    if id_col is None:
        test_ids = [f"TEST_{idx:04d}" for idx in range(len(df))]
    else:
        test_ids = df[id_col].astype(str).tolist()
    texts = df[text_col].fillna("").astype(str).tolist() if text_col is not None else [""] * len(df)
    return pd.DataFrame({"test_id": test_ids, "text": texts, "label": labels})


def csv_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, quoting=csv.QUOTE_ALL, lineterminator="\n").encode("utf-8")


st.set_page_config(page_title="V11 Manipulation Detector", layout="wide")
st.title("V11 Text-First BERT Manipulation Detector")

model_dir = st.sidebar.text_input("Model directory", str(common.DEFAULT_MODEL_DIR))
score_path = st.sidebar.text_input("Scored parquet/CSV", str(common.DEFAULT_PROCESSED / "v11_full_scored.parquet"))
analytics_dir = st.sidebar.text_input("Aggregate analytics directory", str(common.ROOT / "outputs" / "analytics"))

tab_live, tab_batch, tab_dash = st.tabs(["Live tahmin", "Batch tahmin", "Manipülasyon haritası"])

with tab_live:
    text = st.text_area("Jüri metni", height=180)
    if st.button("Tahmin et", type="primary") and text.strip():
        result = predict_text(text, Path(model_dir))
        c1, c2, c3 = st.columns(3)
        c1.metric("Karar", result["decision_tr"])
        c2.metric("Manipülatif skor", f"{result['manipulative_score']:.3f}")
        c3.metric("Organiklik", f"{result['organic_score']:.3f}")
        if result["review_flag"]:
            st.warning("Skor karar eşiğine yakın: manuel review önerilir.")
        st.write("Gerekçeler")
        st.write(result["reasons"])
        st.write("Yakın manuel örnekler")
        st.json(result["nearest_manual_examples"])

with tab_batch:
    st.caption("Beklenen jüri/test formatı: `test_id,text`. Çıktıda `label` 0-1 arası manipülatiflik skorudur.")
    template = pd.DataFrame(
        {
            "test_id": ["TEST_0000", "TEST_0001"],
            "text": [
                "I got a likely phishing message about a fake crypto giveaway.",
                "No transactions on weekends; I will check it again tomorrow.",
            ],
        }
    )
    st.download_button(
        "Örnek input CSV indir",
        csv_download_bytes(template),
        "v11_batch_input_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV yükle: `test_id,text` formatı beklenir", type=["csv"])
    st.caption("Batch çıktısı `test_id,text,label` kolonlarını üretir; `label` 0-1 arası manipülatiflik skorudur.")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.columns = [str(col).strip() for col in df.columns]
        text_col = resolve_column(df, BATCH_TEXT_COLUMNS)
        id_col = resolve_column(df, BATCH_ID_COLUMNS)
        if text_col is None:
            st.error("CSV içinde `text` kolonu yok. Beklenen format: `test_id,text`.")
        else:
            if id_col is None:
                st.warning("`test_id` kolonu bulunamadı; çıktı için TEST_0000 formatında sıra ID üretilecek.")
            if st.button("Batch skorla"):
                results = predict_many(df[text_col].fillna("").astype(str).tolist(), Path(model_dir), batch_size=32)
                output = build_batch_output(df, results)
                st.dataframe(output.head(200), use_container_width=True)
                st.download_button(
                    "Batch tahmin CSV indir",
                    csv_download_bytes(output),
                    "v11_batch_predictions.csv",
                    mime="text/csv",
                )

with tab_dash:
    path = Path(score_path)
    aggregates = Path(analytics_dir)
    if not path.exists():
        st.info("Skorlanmış parquet/CSV bulunamadı; aggregate analytics dosyaları gösteriliyor.")
        summary_path = aggregates / "analytics_summary.json"
        if summary_path.exists():
            summary = common.read_json(summary_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("Skorlanan satır", f"{int(summary.get('rows_scored', 0)):,}")
            c2.metric("Manipülatif oran", f"{summary.get('decision_rates', {}).get('manipulative_rate', 0):.2%}")
            c3.metric("Review oranı", f"{summary.get('decision_rates', {}).get('review_rate', 0):.2%}")
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Platform x dil risk")
            pl_path = aggregates / "platform_language_risk.csv"
            if pl_path.exists():
                st.dataframe(pd.read_csv(pl_path).head(100), use_container_width=True)
            st.subheader("Tema risk dağılımı")
            theme_path = aggregates / "theme_risk.csv"
            if theme_path.exists():
                theme = pd.read_csv(theme_path)
                st.bar_chart(theme.set_index("primary_theme")["mean_manipulative_score"])
        with cols[1]:
            st.subheader("Author risk")
            author_path = aggregates / "author_risk.csv"
            if author_path.exists():
                st.dataframe(pd.read_csv(author_path).head(100), use_container_width=True)
            st.subheader("Zaman trendi")
            trend_path = aggregates / "daily_risk_trend.csv"
            if trend_path.exists():
                trend = pd.read_csv(trend_path)
                if {"date_day", "mean_manipulative_score"}.issubset(trend.columns):
                    st.line_chart(trend.set_index("date_day")["mean_manipulative_score"])
        st.subheader("Yüksek risk örnekleri")
        examples_path = aggregates / "high_risk_examples.csv"
        if examples_path.exists():
            st.dataframe(pd.read_csv(examples_path).head(100), use_container_width=True)
    else:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
        if len(df) > 250_000:
            df = df.sample(250_000, random_state=59)
            st.caption("Dashboard performansı için 250k örnek gösteriliyor.")
        st.metric("Gösterilen satır", f"{len(df):,}")
        if "decision" in df:
            st.bar_chart(df["decision"].value_counts())
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Platform x dil ortalama risk")
            if {"url", "language", "manipulative_score"}.issubset(df.columns):
                heat = (
                    df.groupby(["url", "language"])["manipulative_score"]
                    .mean()
                    .reset_index()
                    .sort_values("manipulative_score", ascending=False)
                    .head(80)
                    .pivot(index="url", columns="language", values="manipulative_score")
                )
                st.dataframe(heat.style.background_gradient(cmap="Reds"), use_container_width=True)
        with cols[1]:
            st.subheader("Tema risk dağılımı")
            if {"primary_theme", "manipulative_score"}.issubset(df.columns):
                theme = df.groupby("primary_theme")["manipulative_score"].mean().sort_values(ascending=False)
                st.bar_chart(theme)
        st.subheader("Author risk")
        if {"author_hash", "manipulative_score"}.issubset(df.columns):
            author = (
                df.groupby("author_hash")
                .agg(rows=("manipulative_score", "size"), mean_risk=("manipulative_score", "mean"), p95_risk=("manipulative_score", lambda s: s.quantile(0.95)))
                .query("rows >= 3")
                .sort_values(["mean_risk", "rows"], ascending=[False, False])
                .head(100)
            )
            st.dataframe(author, use_container_width=True)
        st.subheader("Yüksek risk örnekleri")
        if "manipulative_score" in df:
            st.dataframe(df.sort_values("manipulative_score", ascending=False).head(100), use_container_width=True)
