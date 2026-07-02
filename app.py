import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import cv2
import numpy as np
import streamlit as st
from PIL import Image

import keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Dense

# ══════════════════════════════════════════════════════════════════════════
# Konfigurasi — sama persis dengan notebook training
# ══════════════════════════════════════════════════════════════════════════
IMG_SIZE = (224, 224)

# ══════════════════════════════════════════════════════════════════════════
# Fungsi preprocessing MRI — disalin PERSIS dari notebook training (Cell 20)
# ══════════════════════════════════════════════════════════════════════════
def crop_brain_region(img_bgr, threshold=15, pad=8):
    """Buang background hitam di sekitar citra MRI otak."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    area_ratio = (w * h) / (img_bgr.shape[0] * img_bgr.shape[1])
    if area_ratio < 0.10:
        return img_bgr
    x1 = max(0, x - pad);           y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y1:y2, x1:x2]


def apply_clahe(img_bgr):
    """Perkuat kontras lokal agar fitur tumor lebih menonjol."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def mri_preprocess(pil_image: Image.Image, target_size=IMG_SIZE) -> np.ndarray:
    """Pipeline preprocessing identik dengan training."""
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = crop_brain_region(img_bgr)
    img_bgr = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    img_bgr = apply_clahe(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(preprocess_input(img_rgb), axis=0)


def get_preprocessed_preview(pil_image: Image.Image) -> np.ndarray:
    """Gambar hasil crop + CLAHE untuk ditampilkan di UI."""
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = crop_brain_region(img_bgr)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_bgr = apply_clahe(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════
# Out-of-Distribution (OOD) Detection
# Menggabungkan beberapa sinyal heuristik agar deteksi citra non-MRI lebih
# robust dibanding satu aturan tunggal. Tidak mengubah arsitektur model —
# hanya lapisan validasi tambahan sebelum hasil klasifikasi ditampilkan.
# ══════════════════════════════════════════════════════════════════════════
OOD_THRESHOLDS = {
    "saturation_max":      8.0,    # rata-rata saturasi HSV (0-255). MRI asli ~0.
    "dark_bg_ratio_min":   0.03,   # MRI biasanya punya latar hitam yang cukup luas
    "brain_area_ratio_min": 0.08,  # objek/foreground harus punya area signifikan
    "max_confidence_min": 0.50,    # keyakinan model minimal terhadap 1 kelas
    "entropy_max":         1.20,   # log(4) ≈ 1.386 -> makin tinggi makin ambigu
}


def _saturation_score(img_bgr: np.ndarray) -> float:
    """
    Rata-rata saturasi HSV. MRI grayscale asli punya saturasi ~0 (bahkan
    setelah kompresi JPEG). Metrik ini jauh lebih tajam dibanding sekadar
    membandingkan rata-rata selisih channel RGB, karena foto berwarna dengan
    palet pudar/pastel (mis. foto langit, awan, es) bisa punya selisih RGB
    kecil tapi saturasinya tetap jelas berbeda dari citra grayscale asli.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    return float(saturation.mean())


def _dark_background_ratio(gray_img: np.ndarray, threshold: int = 15) -> float:
    """Proporsi piksel yang sangat gelap (latar belakang khas MRI)."""
    return float(np.mean(gray_img < threshold))


def _brain_area_ratio(img_bgr: np.ndarray, threshold: int = 15) -> float:
    """Rasio area bounding box kontur terbesar terhadap seluruh citra."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    _, _, w, h = cv2.boundingRect(largest)
    return (w * h) / (img_bgr.shape[0] * img_bgr.shape[1])


def _prediction_entropy(prediction: np.ndarray) -> float:
    """Entropi Shannon dari distribusi probabilitas softmax."""
    eps = 1e-8
    p = np.clip(prediction, eps, 1.0)
    return float(-np.sum(p * np.log(p)))


def validate_mri_image(pil_image: Image.Image, prediction: np.ndarray):
    """
    Deteksi OOD berbasis kombinasi sinyal citra + kepercayaan model.

    Returns
    -------
    is_valid : bool
        True jika citra dianggap kemungkinan besar MRI otak yang valid.
    reasons : list[str]
        Daftar alasan (dalam Bahasa Indonesia) jika citra ditolak.
    scores : dict
        Nilai mentah tiap sinyal, untuk keperluan debugging / laporan skripsi.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    saturation_score = _saturation_score(img_bgr)
    dark_bg_ratio    = _dark_background_ratio(gray)
    brain_area_ratio = _brain_area_ratio(img_bgr)
    max_confidence   = float(np.max(prediction))
    entropy          = _prediction_entropy(prediction)

    reasons = []
    if saturation_score > OOD_THRESHOLDS["saturation_max"]:
        reasons.append("Citra tampak berwarna, bukan grayscale khas MRI")
    if dark_bg_ratio < OOD_THRESHOLDS["dark_bg_ratio_min"]:
        reasons.append("Tidak ada area gelap latar belakang khas citra MRI")
    if brain_area_ratio < OOD_THRESHOLDS["brain_area_ratio_min"]:
        reasons.append("Tidak ditemukan objek/struktur menyerupai otak")
    if max_confidence < OOD_THRESHOLDS["max_confidence_min"]:
        reasons.append(f"Model tidak yakin pada kelas manapun (confidence {max_confidence*100:.1f}%)")
    if entropy > OOD_THRESHOLDS["entropy_max"]:
        reasons.append("Distribusi probabilitas antar kelas terlalu merata (ambigu)")

    scores = {
        "saturation_score": saturation_score,
        "dark_bg_ratio": dark_bg_ratio,
        "brain_area_ratio": brain_area_ratio,
        "max_confidence": max_confidence,
        "entropy": entropy,
    }
    return len(reasons) == 0, reasons, scores

# FIX error: Unrecognized keyword arguments Dense: quantization_config
_original_dense_from_config = Dense.from_config

def _patched_dense_from_config(config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)

Dense.from_config = classmethod(lambda cls, config: _patched_dense_from_config(config))
# ══════════════════════════════════════════════════════════════════════════
# Load model & label
# FIX: gunakan keras.saving.load_model (keras 3.x API)
#      bukan keras.models.load_model yang berperilaku berbeda di keras 3
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat model…")
def load_resources():
    try:
        model = keras.saving.load_model(
            "tumor_classifier_final.keras",
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

    with open("label_order.pkl", "rb") as f:
        label_order = pickle.load(f)

    with open("label_display.pkl", "rb") as f:
        label_display = pickle.load(f)

    return model, label_order, label_display


# ══════════════════════════════════════════════════════════════════════════
# Konfigurasi halaman
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Klasifikasi Tumor Otak MRI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# Custom styling — Dark theme
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; }

    #MainMenu, header, footer { visibility: hidden; }

    /* Dark background */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1B1035 0%, #0B0E1A 45%, #08090F 100%);
        color: #E5E7EB;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1220px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12101F 0%, #0B0E1A 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #D1D5DB !important; }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #4C1D95 0%, #7C3AED 45%, #C026D3 100%);
        border-radius: 24px;
        padding: 2.6rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 15px 45px rgba(124, 58, 237, 0.35);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -60px; top: -80px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.15), transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.02rem;
        margin: 0;
        max-width: 640px;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(6px);
        color: #fff;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Stat pills under hero */
    .stat-row { display: flex; gap: 0.9rem; margin-bottom: 2rem; flex-wrap: wrap; }
    .stat-pill {
        flex: 1;
        min-width: 150px;
        background: linear-gradient(145deg, #14172A, #10121F);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1rem 1.1rem;
    }
    .stat-pill .num { font-size: 1.3rem; font-weight: 800; font-family: 'Poppins', sans-serif; color: #fff; }
    .stat-pill .lbl { font-size: 0.78rem; color: #9CA3AF; margin-top: 2px; }

    /* Section cards */
    .card {
        background: linear-gradient(160deg, #14182B 0%, #0F1220 100%);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(255,255,255,0.07);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        height: 100%;
    }
    .card h4 {
        margin-top: 0; margin-bottom: 0.9rem;
        font-size: 1.02rem; font-weight: 600; color: #F3F4F6;
    }
    .card img { border-radius: 12px; }

    /* Result hero card */
    .result-card {
        border-radius: 20px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
        color: #fff;
        position: relative;
        overflow: hidden;
    }
    .result-card .label { font-size: 0.82rem; opacity: 0.85; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px; }
    .result-card .title { font-size: 1.6rem; font-weight: 800; font-family: 'Poppins', sans-serif; margin-bottom: 6px; }
    .result-card .sub { font-size: 0.92rem; opacity: 0.92; }
    .result-good    { background: linear-gradient(135deg,#065F46,#10B981); box-shadow: 0 10px 30px rgba(16,185,129,0.25); }
    .result-medium  { background: linear-gradient(135deg,#92400E,#F59E0B); box-shadow: 0 10px 30px rgba(245,158,11,0.25); }
    .result-low     { background: linear-gradient(135deg,#7F1D1D,#EF4444); box-shadow: 0 10px 30px rgba(239,68,68,0.25); }

    /* Confidence donut */
    .donut-wrap { display:flex; align-items:center; gap: 1rem; margin-bottom: 1.1rem; }
    .donut {
        width: 84px; height: 84px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        background: conic-gradient(var(--donut-color) calc(var(--pct) * 1%), #232741 0);
        flex-shrink: 0;
    }
    .donut-inner {
        width: 64px; height: 64px; border-radius: 50%;
        background: #0F1220;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 1rem; color: #fff; font-family: 'Poppins', sans-serif;
    }
    .donut-text .t1 { font-size: 0.82rem; color: #9CA3AF; font-weight: 500; }
    .donut-text .t2 { font-size: 0.95rem; color: #E5E7EB; font-weight: 600; }

    /* Probability bars */
    .prob-row { margin-bottom: 12px; }
    .prob-top { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 4px; color: #D1D5DB; font-weight: 500; }
    .prob-track { background: #1E2235; border-radius: 8px; height: 10px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg,#818CF8,#E879F9); }

    /* Class chips on landing */
    .chip-card {
        border-radius: 18px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(160deg, #14182B 0%, #0F1220 100%);
        height: 100%;
        transition: transform 0.15s ease;
    }
    .chip-card .icon { font-size: 1.7rem; margin-bottom: 6px; }
    .chip-card .name { font-weight: 700; font-size: 0.95rem; color: #F3F4F6; margin-bottom: 3px; }
    .chip-card .desc { font-size: 0.8rem; color: #9CA3AF; }

    /* How it works steps */
    .step-card {
        border-radius: 18px;
        padding: 1.2rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.07);
        background: linear-gradient(160deg, #14182B 0%, #0F1220 100%);
        height: 100%;
    }
    .step-card .num {
        width: 30px; height: 30px; border-radius: 50%;
        background: linear-gradient(135deg,#7C3AED,#C026D3);
        color: #fff; display:flex; align-items:center; justify-content:center;
        font-weight: 700; font-size: 0.9rem; margin-bottom: 0.7rem;
    }
    .step-card .stitle { font-weight: 700; color: #F3F4F6; font-size: 0.92rem; margin-bottom: 3px; }
    .step-card .sdesc { font-size: 0.8rem; color: #9CA3AF; }

    .disclaimer {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        font-size: 0.9rem;
        color: #DDD6FE;
        margin-top: 1rem;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(167, 139, 250, 0.4);
        border-radius: 16px;
        padding: 0.6rem;
        background: rgba(124, 58, 237, 0.05);
    }
    div[data-testid="stFileUploader"] section { background: transparent; }
    div[data-testid="stFileUploaderDropzoneInstructions"] * { color: #D1D5DB !important; }

    /* History items in sidebar */
    .hist-item {
        background: #14182B;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.55rem;
    }
    .hist-item .hname { font-weight: 700; font-size: 0.85rem; color: #F3F4F6; }
    .hist-item .hmeta { font-size: 0.72rem; color: #9CA3AF; }

    .footer-note {
        text-align: center;
        color: #6B7280;
        font-size: 0.82rem;
        margin-top: 2.5rem;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg,#7C3AED,#C026D3);
        color: #fff; border: none; border-radius: 10px; font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        filter: brightness(1.1);
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []

RESULT_TIER_COLOR = {
    "result-good": "#10B981",
    "result-medium": "#F59E0B",
    "result-low": "#EF4444",
}

# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 Brain MRI Classifier")
    st.caption("Alat bantu skrining citra MRI otak")
    st.markdown("---")
    st.markdown("#### 🕘 Riwayat Analisis")

    if st.session_state.history:
        if st.button("🗑️ Bersihkan Riwayat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        for item in reversed(st.session_state.history[-8:]):
            st.markdown(f"""
            <div class="hist-item">
                <div class="hname">{item['label']}</div>
                <div class="hmeta">{item['confidence']:.1f}% keyakinan · {item['time']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("Belum ada riwayat analisis pada sesi ini.")

    st.markdown("---")
    st.caption("⚠️ Hasil klasifikasi bersifat bantu skrining dan bukan pengganti diagnosis dokter.")

# ══════════════════════════════════════════════════════════════════════════
# Hero
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="badge">🧠 MRI BRAIN ANALYSIS</div>
    <h1>Klasifikasi Tumor Otak MRI</h1>
    <p>Unggah citra MRI otak dan dapatkan hasil klasifikasi secara instan,
    dilengkapi tingkat dan rincian probabilitas tiap jenis penyakit.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-pill"><div class="num">4</div><div class="lbl">Kategori Terdeteksi</div></div>
    <div class="stat-pill"><div class="num">⚡ Instan</div><div class="lbl">Waktu Analisis</div></div>
    <div class="stat-pill"><div class="num">🖱️ 1-Klik</div><div class="lbl">Upload & Lihat Hasil</div></div>
    <div class="stat-pill"><div class="num">🔒 Lokal</div><div class="lbl">Diproses di Sesi Anda</div></div>
</div>
""", unsafe_allow_html=True)

model, label_order, label_display = load_resources()

# ══════════════════════════════════════════════════════════════════════════
# Upload & Prediksi
# ══════════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "📤 Upload citra MRI otak (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        pil_image     = Image.open(uploaded_file)
        input_tensor  = mri_preprocess(pil_image)
        preview_img   = get_preprocessed_preview(pil_image)

        prediction    = model.predict(input_tensor, verbose=0)[0]
        pred_index    = int(np.argmax(prediction))
        pred_label    = label_order[pred_index]
        pred_display  = label_display[pred_label]
        confidence    = float(prediction[pred_index]) * 100

        is_valid_mri, ood_reasons, ood_scores = validate_mri_image(pil_image, prediction)

        st.write("")
        col1, col2, col3 = st.columns([1, 1, 1.2], gap="medium")

        with col1:
            st.markdown('<div class="card"><h4>🖼️ Citra Asli</h4>', unsafe_allow_html=True)
            st.image(pil_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><h4>✨ Citra yang Dianalisis</h4>', unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="card"><h4>📊 Hasil Klasifikasi</h4>', unsafe_allow_html=True)

            if not is_valid_mri:
                st.markdown('</div>', unsafe_allow_html=True)
                st.error("🚫 Hasil Ditolak ini bukan MRI Scan")
                st.info(
                    "ℹ️ Silakan upload ulang citra MRI otak "
                )
                st.stop()

            if confidence >= 80:
                tier_class, tier_note = "result-good", "Tinggi"
            elif confidence >= 60:
                tier_class, tier_note = "result-medium", "Sedang"
            else:
                tier_class, tier_note = "result-low", "Rendah"

            st.markdown(f"""
            <div class="result-card {tier_class}">
                <div class="label">HASIL PREDIKSI</div>
                <div class="title">{pred_display}</div>
                <div class="sub">{tier_note} · {confidence:.1f}% </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="donut-wrap">
                <div class="donut" style="--pct:{confidence:.1f}; --donut-color:{RESULT_TIER_COLOR[tier_class]};">
                    <div class="donut-inner">{confidence:.0f}%</div>
                </div>
                <div class="donut-text">
                    <div class="t1">Hasil Probabilitas</div>
                    <div class="t2">{tier_note}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Probabilitas semua kelas**")
            sorted_pairs = sorted(
                zip(label_order, prediction),
                key=lambda x: x[1], reverse=True
            )
            rows_html = ""
            for label, prob in sorted_pairs:
                pct  = float(prob) * 100
                flag = "✅ " if label == pred_label else ""
                rows_html += f"""
                <div class="prob-row">
                    <div class="prob-top"><span>{flag}{label_display[label]}</span><span>{pct:.1f}%</span></div>
                    <div class="prob-track"><div class="prob-fill" style="width:{pct}%;"></div></div>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

            if confidence < 60:
                st.markdown('<div class="disclaimer">⚠️ Hasil Probabilitas di bawah 60% — disarankan konfirmasi ke fasilitas kesehatan.</div>', unsafe_allow_html=True)
            elif confidence < 80:
                st.markdown('<div class="disclaimer">⚠️ Keyakinan 60–80% — disarankan konfirmasi medis lebih lanjut.</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Simpan ke riwayat sesi (sekali per file baru)
        from datetime import datetime
        last_key = st.session_state.get("_last_file_id")
        current_key = f"{uploaded_file.name}-{uploaded_file.size}"
        if last_key != current_key:
            st.session_state.history.append({
                "label": pred_display,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S"),
            })
            st.session_state._last_file_id = current_key

        st.write("")
        report_text = (
            f"HASIL KLASIFIKASI MRI OTAK\n"
            f"============================\n"
            f"File            : {uploaded_file.name}\n"
            f"Prediksi        : {pred_display}\n"
            f"Tingkat Keyakinan: {confidence:.2f}%\n\n"
            f"Rincian Probabilitas:\n"
        )
        for label, prob in sorted_pairs:
            report_text += f"- {label_display[label]}: {float(prob)*100:.2f}%\n"
        report_text += (
            "\nCatatan: Hasil ini hanya alat bantu skrining, bukan pengganti "
            "diagnosis medis profesional.\n"
        )

        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            st.download_button(
                "⬇️ Unduh Ringkasan Hasil",
                data=report_text,
                file_name=f"hasil_klasifikasi_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown(
            '<div class="disclaimer">ℹ️ Sistem ini hanya alat bantu klasifikasi citra MRI '
            'dan bukan pengganti diagnosis dokter.</div>',
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Error saat memproses gambar: {e}")

else:
    st.write("")
    st.markdown("#### Kelas yang Dapat Dikenali")
    cols = st.columns(4, gap="medium")
    kelas = [
        ("🔴", "Glioma Tumor",     "Tumor ganas dari sel glial"),
        ("🟡", "Meningioma Tumor", "Tumor pada selaput pembungkus otak"),
        ("🟢", "No Tumor",         "Tidak terdeteksi tumor"),
        ("🔵", "Pituitary Tumor",  "Tumor pada kelenjar pituitari"),
    ]
    for col, (icon, nama, desc) in zip(cols, kelas):
        with col:
            st.markdown(f"""
            <div class="chip-card">
                <div class="icon">{icon}</div>
                <div class="name">{nama}</div>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Cara Penggunaan")
    scols = st.columns(3, gap="medium")
    steps = [
        ("1", "Upload Citra", "Pilih file MRI otak berformat JPG, JPEG, atau PNG."),
        ("2", "Tunggu Proses", "Sistem menganalisis citra secara otomatis dan instan."),
        ("3", "Lihat Hasil", "Kategori, tingkat keyakinan, dan rincian probabilitas ditampilkan."),
    ]
    for col, (num, title, desc) in zip(scols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="num">{num}</div>
                <div class="stitle">{title}</div>
                <div class="sdesc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer-note">Dashboard Klasifikasi Tumor Otak MRI · '
        'Alat bantu skrining, bukan pengganti diagnosis medis</div>',
        unsafe_allow_html=True
    )
