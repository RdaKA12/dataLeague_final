# V11 Jury Delivery Report

## 1. Güvenilirlik / Organiklik Skoru Algoritması

- Ana model: GPU ile eğitilmiş text-only `xlm-roberta-base` binary classifier.
- Her içerik için `manipulative_score` ve `organic_score = 1 - manipulative_score` üretilir.
- Canlı jüri tahmini yalnız metne dayanır; metadata model kararına feature olarak girmez.
- Author seviyesi risk, içerik skorlarının agregasyonu olarak dashboard tarafında hesaplanır.

### Model metrikleri

| Set | ROC-AUC | PR-AUC | M Precision | M Recall | M F1 |
|---|---:|---:|---:|---:|---:|
| Holdout | 0.9248 | 0.7059 | 0.4966 | 0.8130 | 0.6166 |
| Text-only Qwen validation | 0.9430 | 0.8919 | 0.6921 | 0.8992 | 0.7822 |

## 2. Manipülasyon Haritası / Dashboard

- Skorlanan satır: `5,004,813`
- Ortalama manipülatif skor: `0.1582`
- Manipulatif karar oranı: `19.13%`
- Review oranı: `4.30%`
- Dashboard artefactleri platform, dil, tema, zaman, kullanıcı ve duplicate cluster kırılımlarını üretir.

## 3. Canlı Tahmin Modeli

- CLI: `python version11_bert\scripts\v11_inference.py --text "..." --json`
- Dashboard: `streamlit run version11_bert\scripts\v11_dashboard.py`
- Çıktı: karar, manipülatif skor, organiklik skoru, review flag, gerekçeler ve yakın manuel örnekler.

## 4. Rubric’e Göre Durum

- Analitik derinlik: platform/dil/tema/zaman/author/duplicate cluster analizleri üretildi.
- Model ve canlı tahmin: external text-only validation üzerinde yüksek recall ve PR-AUC; CLI çalışıyor.
- Dashboard ve sunum: Streamlit app aynı bundle ve skor dosyalarını kullanıyor.
- Teknik uygulanabilirlik: GPU eğitim, resumable full scoring, shard combine ve hata durumunda resume desteği var.

## 5. Geliştirilebilir Yerler

- False positive azaltma: `final_low` ve `background` segmentlerindeki FP örnekleriyle threshold veya calibration iyileştirilebilir.
- Açıklanabilirlik: token attribution için Integrated Gradients eklenirse gerekçeler daha model-içi olur.
- Author/network modeli: canlı text-only karar korunarak dashboard’a ayrı metadata/context ranker eklenebilir.
- Aktif öğrenme: V11’in yanlış pozitif/yanlış negatiflerinden 2-5k yeni manuel audit turu performansı yükseltir.
- Full scoring periyodikleştirilebilir: yeni veri geldiğinde sadece yeni row-group/shard skorlanır.

## 6. Üretilen Artefactler

- `platform_language_risk`: `D:\Projects\datalig\version11_bert\outputs\analytics\platform_language_risk.csv`
- `platform_risk`: `D:\Projects\datalig\version11_bert\outputs\analytics\platform_risk.csv`
- `language_risk`: `D:\Projects\datalig\version11_bert\outputs\analytics\language_risk.csv`
- `theme_risk`: `D:\Projects\datalig\version11_bert\outputs\analytics\theme_risk.csv`
- `daily_risk_trend`: `D:\Projects\datalig\version11_bert\outputs\analytics\daily_risk_trend.csv`
- `author_risk`: `D:\Projects\datalig\version11_bert\outputs\analytics\author_risk.csv`
- `high_risk_examples`: `D:\Projects\datalig\version11_bert\outputs\analytics\high_risk_examples.csv`
- `high_risk_duplicate_clusters`: `D:\Projects\datalig\version11_bert\outputs\analytics\high_risk_duplicate_clusters.csv`
