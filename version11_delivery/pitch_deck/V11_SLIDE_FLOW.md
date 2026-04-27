# V11 Pitch Deck - Slayt Akışı

## 1. Kapak
Amaç: V11'in ne olduğunu tek cümlede vermek.
Bileşenler: 5M skorlanan içerik, external ROC-AUC, M recall, DataLeague final bağlamı.

## 2. İstenen Üç Çıktı
Amaç: README'deki üç teslimatı aynı sistemde karşıladığımızı göstermek.
Bileşenler: Organiklik skoru, manipülasyon haritası, canlı inference.

## 3. Mimari
Amaç: Canlı kararın text-only olduğunu, metadata'nın dashboard analitiğinde kullanıldığını açıklamak.
Bileşenler: 60k manuel M/O -> XLM-RoBERTa -> threshold -> 5M scoring -> dashboard.

## 4. Model Başarısı
Amaç: Modelin çalıştığını ve manipülatifi kaçırmama yaklaşımını kanıtlamak.
Bileşenler: Holdout/external ROC-AUC, PR-AUC, M recall, M F1; confusion matrix.
Holdout ROC-AUC: 0.9248
External ROC-AUC: 0.9430
External M recall: 0.8992

## 5. 5M Full Scoring
Amaç: Sadece örneklem değil, tüm veri üzerinde skor üretildiğini göstermek.
Bileşenler: 5,004,813 satır, manipülatif karar oranı 19.13%, review oranı 4.30%.

## 6. Platform x Dil Haritası
Amaç: Dashboard puanını yükselten karar destek bileşenini göstermek.
Bileşenler: yüksek hacimli platform/dil heatmap; x.com/ar ve YouTube/en bulguları.

## 7. Tema ve Zaman
Amaç: Manipülasyonun konu ve zaman boyutunu göstermek.
Bileşenler: Cryptocurrency, Politics, Investing riskleri; günlük manipülatif oran trendi.

## 8. Author ve Duplicate Ağları
Amaç: Bot/spam/anomali örüntülerini yarışma problemiyle ilişkilendirmek.
Bileşenler: top risky authors, yüksek risk duplicate cluster örnekleri.

## 9. Canlı Demo
Amaç: Jürinin vereceği unseen metni nasıl çalıştıracağımızı göstermek.
Bileşenler: CLI komutu, Streamlit demo, skor, karar, review flag, gerekçeler.

## 10. Rubric Haritası
Amaç: Her puan kategorisini somut artefact ile bağlamak.
Bileşenler: analitik derinlik, model+canlı tahmin, dashboard+sunum, teknik uygulanabilirlik.
