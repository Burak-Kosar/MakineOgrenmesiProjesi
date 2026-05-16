import streamlit as st
import pandas as pd
import joblib

# 1. Sayfa Yapılandırması ve Başlık
st.set_page_config(page_title="Kalp Hastalığı Risk Tahmini", page_icon="🫀", layout="centered")

st.title("🫀 Kalp Hastalığı Risk Tahmin Sistemi")
st.markdown("""
Bu sistem, hastanın belirli klinik bulgularını analiz ederek **Naive Bayes** makine öğrenmesi modeli ile kalp hastalığı riskini tahmin eder.
Lütfen sol taraftaki menüden hasta değerlerini girip 'Tahmin Et' butonuna basınız.
""")

# 2. Modelleri Yükleme Fonksiyonu (@st.cache_resource ile modeli sadece 1 kere yüklüyoruz, sunucuyu yormuyoruz)
@st.cache_resource
def load_models():
    # Modeller klasörünün app.py ile aynı dizinde olduğundan emin ol
    model = joblib.load('modeller/nb_model.pkl')
    scaler = joblib.load('modeller/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error("Modeller yüklenemedi! Lütfen 'modeller' klasörünün ve içindeki .pkl dosyalarının doğru yerde olduğundan emin olun.")
    st.stop()

# 3. Sol Menü (Kullanıcı Veri Girişi)
st.sidebar.header("📋 Hasta Bulgularını Girin")

# Seçtiğimiz en iyi 5 özellik için giriş alanları
thalach = st.sidebar.slider("Maksimum Kalp Hızı (thalach)", min_value=70, max_value=210, value=150)
ca = st.sidebar.selectbox("Renkli Floroskopi ile Görülen Damar Sayısı (ca)", options=[0.0, 1.0, 2.0, 3.0])
thal = st.sidebar.selectbox("Talasemi (thal)", options=[3.0, 6.0, 7.0], format_func=lambda x: "Normal (3.0)" if x==3.0 else ("Sabit Hata (6.0)" if x==6.0 else "Tersinir Hata (7.0)"))
cp = st.sidebar.selectbox("Göğüs Ağrısı Tipi (cp)", options=[1.0, 2.0, 3.0, 4.0], format_func=lambda x: "Tipik Anjina (1.0)" if x==1.0 else ("Atipik Anjina (2.0)" if x==2.0 else ("Anjinal Olmayan (3.0)" if x==3.0 else "Asemptomatik (4.0)")))
oldpeak = st.sidebar.slider("Egzersize Bağlı ST Çökmesi (oldpeak)", min_value=0.0, max_value=6.2, value=1.0, step=0.1)

# 4. Tahmin Butonu ve Mantığı
if st.sidebar.button("Tahmin Et 🔍"):
    
    # Kullanıcının girdiği verileri sözlükte topluyoruz
    user_data = {
        'thalach': thalach,
        'ca': ca,
        'thal': thal,
        'cp': cp,
        'oldpeak': oldpeak
    }
    
    # Kodu yazarken dikkat etmemiz gereken kritik nokta: 
    # Scaler aracımız Colab'de 13 sütunun tamamı ile eğitilmişti. 
    # Hata vermemesi için 13 sütunluk boş bir şablon açıp, bizim 5 değerimizi içine yerleştiriyoruz.
    orijinal_kolonlar = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame(0, index=[0], columns=orijinal_kolonlar)
    
    for col, val in user_data.items():
        input_df[col] = val
        
    # Veriyi 0-1 arasına ölçeklendiriyoruz
    scaled_data = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_data, columns=orijinal_kolonlar)
    
    # Modele sadece beklediği 5 özelliği (Eğitim sırasıyla) veriyoruz
    en_iyi_ozellikler = ['thalach', 'ca', 'thal', 'cp', 'oldpeak']
    final_input = scaled_df[en_iyi_ozellikler]
    
    # Yapay Zeka Tahmini Yapıyor
    tahmin_sonucu = model.predict(final_input)[0]
    tahmin_olasiligi = model.predict_proba(final_input)[0]
    
    # 5. Sonuçları Ekranda Gösterme
    st.markdown("---")
    st.subheader("💡 Analiz Sonucu")
    
    if tahmin_sonucu == 1:
        st.error(f"⚠️ **RİSKLİ DURUM:** Modelimiz, girilen bulgulara göre hastada kalp rahatsızlığı olma ihtimali tespit etti.")
        st.warning(f"Hastalık Olasılığı: **% {tahmin_olasiligi[1]*100:.2f}**")
    else:
        st.success(f"✅ **DÜŞÜK RİSK:** Modelimiz, girilen bulgulara göre hastada kalp rahatsızlığı bulgusuna rastlamadı.")
        st.info(f"Sağlıklı Olma Olasılığı: **% {tahmin_olasiligi[0]*100:.2f}**")