# Kalp Hastalığı Tahmin Projesi

BIM 322 Makine Öğrenmesi ve Uygulamaları dersi proje ödevi.

## Proje Hakkında

UCI Cleveland veri seti kullanılarak kalp hastalığı riski tahmin eden bir makine öğrenmesi uygulaması.

## Kullanılan Modeller

- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest

## Klasör Yapısı

```
Makine_Projesi/
├── data/                  # Ham veri seti
├── notebooks/             # Araştırma ve model eğitim kodları
├── modeller/              # Eğitilmiş model dosyaları (.pkl)
├── sunum/                 # Proje sunumu (.pptx)
├── app.py                 # Streamlit web arayüzü
└── requirements.txt       # Gerekli kütüphaneler
```

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```
