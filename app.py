import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Tomat",
    layout="centered"
)
# ============================================
# MODERN DARK UI
# ============================================
st.markdown("""
<style>

/* IMPORT FONT */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;600;700&display=swap');

/* MAIN APP */
.stApp {

    background: #050816;
    color: white;
    font-family: 'Inter', sans-serif;
}

/* CONTAINER */
.block-container {

    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* TEXT */
h1,h2,h3,h4,h5,h6 {

    color: white !important;
    font-family: 'Poppins', sans-serif;
}

p,label,div,span {

    color: #d6d6d6 !important;
}

/* TITLE */
h1 {

    font-size: 42px !important;
    font-weight: 700 !important;
    margin-bottom: 10px;
}

/* SUBTITLE */
.subtitle {

    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 30px;
}

/* LINE */
hr {

    border: 1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR */
[data-testid="stSidebar"] {

    background-color: #030712;
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {

    color: white !important;
}

/* SELECTBOX */
.stSelectbox label {

    color: white !important;
    font-weight: 600;
}

/* SELECTBOX FIELD */
.stSelectbox div[data-baseweb="select"] > div {

    background-color: #111827 !important;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    color: white !important;
}

/* SELECTBOX TEXT */
.stSelectbox div[data-baseweb="select"] span {

    color: white !important;
}

/* DROPDOWN */
div[data-baseweb="popover"] * {

    color: white !important;
    background-color: #111827 !important;
}

/* EXPANDER */
.streamlit-expanderHeader {

    background-color: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* EXPANDER CONTENT */
.streamlit-expanderContent {

    background-color: rgba(255,255,255,0.02);
    border-radius: 0 0 10px 10px;
    border: 1px solid rgba(255,255,255,0.04);
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {

    background-color: rgba(255,255,255,0.04);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.05);
    padding: 10px;
}

/* BUTTON */
.stButton button {

    background-color: #111827;
    color: white;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}

/* IMAGE */
img {

    border-radius: 16px;
}

/* CHART */
canvas {

    background-color: rgba(255,255,255,0.01);
    border-radius: 12px;
}

/* MOBILE */
@media (max-width: 768px) {

    h1 {

        font-size: 30px !important;
    }

    .block-container {

        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)
# ============================================
# MODERN HEADER
# ============================================
st.markdown("""
<h1 style='
text-align:center;
font-size:42px;
font-weight:700;
background: linear-gradient(
90deg,#4ade80,#22c55e,#16a34a
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
'>
🌿 Tomato AI Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='
text-align:center;
color:#9ca3af;
font-size:16px;
margin-bottom:30px;
'>
Sistem klasifikasi penyakit daun tomat berbasis Deep Learning
menggunakan MobileNetV2 dan EfficientNetB0
pada berbagai skenario pelatihan.
</p>
""", unsafe_allow_html=True)

# ============================================
# DASHBOARD METRICS
# ============================================
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div style="
    background: rgba(255,255,255,0.04);
    padding:20px;
    border-radius:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.06);
    ">
    <h3>📊 Dataset</h3>
    <h2>10</h2>
    <p>Kelas Penyakit</p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div style="
    background: rgba(255,255,255,0.04);
    padding:20px;
    border-radius:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.06);
    ">
    <h3>🧠 CNN Model</h3>
    <h2>2</h2>
    <p>Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div style="
    background: rgba(255,255,255,0.04);
    padding:20px;
    border-radius:16px;
    text-align:center;
    border:1px solid rgba(255,255,255,0.06);
    ">
    <h3>⚡ Skenario</h3>
    <h2>{variant}</h2>
    <p>Model Training</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================
# TIPS
# ============================================
st.info("""
📸 Tips Upload:
- Gunakan gambar daun yang jelas
- Hindari blur
- Fokus pada daun
- Gunakan pencahayaan cukup
""")

st.markdown("---")

# ============================================
# INFORMASI PENYAKIT
# ============================================
st.subheader("📚 Informasi Kelas Penyakit")

with st.expander("🟤 Bacterial Spot"):
    st.write("""
    Penyakit bercak bakteri yang menyebabkan bercak kecil coklat gelap pada daun.
    
    Penanganan:
    - Gunakan benih sehat
    - Hindari kelembaban tinggi
    - Gunakan bakterisida
    """)

with st.expander("🟠 Early Blight"):
    st.write("""
    Ditandai bercak melingkar seperti cincin target pada daun.
    
    Penanganan:
    - Buang daun terinfeksi
    - Gunakan fungisida
    """)

with st.expander("⚫ Late Blight"):
    st.write("""
    Penyakit dengan bercak gelap basah yang cepat menyebar.
    
    Penanganan:
    - Kurangi kelembaban
    - Gunakan fungisida
    """)

with st.expander("🟢 Leaf Mold"):
    st.write("""
    Menyebabkan bercak kekuningan dan pertumbuhan jamur.
    
    Penanganan:
    - Perbaiki sirkulasi udara
    """)

with st.expander("🟡 Septoria Leaf Spot"):
    st.write("""
    Bercak kecil abu-abu dengan tepi gelap.
    
    Penanganan:
    - Buang daun terinfeksi
    """)

with st.expander("🕷 Spider Mites"):
    st.write("""
    Serangan tungau yang merusak jaringan daun.
    
    Penanganan:
    - Gunakan pestisida tungau
    """)

with st.expander("🎯 Target Spot"):
    st.write("""
    Bercak melingkar menyerupai target.
    
    Penanganan:
    - Gunakan fungisida
    """)

with st.expander("🦠 Tomato Mosaic Virus"):
    st.write("""
    Virus yang menyebabkan pola mosaik pada daun.
    
    Penanganan:
    - Cabut tanaman terinfeksi
    """)

with st.expander("🌿 Tomato Yellow Leaf Curl Virus"):
    st.write("""
    Virus yang menyebabkan daun menguning dan melengkung.
    
    Penanganan:
    - Kendalikan kutu putih
    """)

with st.expander("✅ Healthy"):
    st.write("""
    Daun tomat sehat tanpa gejala penyakit.
    """)

st.markdown("---")

# ============================================
# MODEL SELECTOR
# ============================================
variant = st.selectbox(
    "Pilih Skenario Model",
    ["FF", "FT10", "FT20", "FT30"]
)

# ============================================
# FILE UPLOADER
# ============================================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Daun Tomat",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# INFERENCE
# ============================================
if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.subheader("🖼️ Gambar Input")

        col_l, col_c, col_r = st.columns([1,2,1])

        with col_c:

            st.image(
                image,
                use_container_width=True
            )

        st.markdown("---")

        # ============================================
        # PREPROCESS
        # ============================================
        x_mn = preprocess_mobilenet(image)
        x_ef = preprocess_efficientnet(image)

        # ============================================
        # LOAD MODEL
        # ============================================
        with st.spinner(
            "🤖 AI sedang menganalisis gambar..."
        ):

            model_mn = load_single_model(
                "MobileNetV2",
                variant
            )

            model_ef = load_single_model(
                "EfficientNetB0",
                variant
            )

            pred_mn = model_mn.predict(
                x_mn,
                verbose=0
            )[0]

            pred_ef = model_ef.predict(
                x_ef,
                verbose=0
            )[0]

        # ============================================
        # CONFIDENCE
        # ============================================
        conf_mn = float(np.max(pred_mn))
        conf_ef = float(np.max(pred_ef))

        idx_mn = int(np.argmax(pred_mn))
        idx_ef = int(np.argmax(pred_ef))

        threshold = CONF_THRESHOLDS[variant]

        # ============================================
        # CONFIDENCE GATE
        # ============================================
        if conf_mn < threshold or conf_ef < threshold:

            st.error("""
            Gambar tidak valid atau kualitas terlalu rendah.
            Silakan upload ulang gambar daun tomat.
            """)

        else:

            st.markdown("---")

            st.subheader(
                f"📊 Hasil Prediksi ({variant})"
            )

            col1, col2 = st.columns(2)

            # ============================================
            # MOBILENETV2
            # ============================================
            with col1:

                st.markdown("""
                <div style="
                background: rgba(255,255,255,0.04);
                padding:20px;
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.06);
                ">
                """, unsafe_allow_html=True)

                st.markdown("### MobileNetV2")

                st.success(
                    f"Prediksi: {CLASS_NAMES[idx_mn]}"
                )

                st.write(
                    f"Confidence: {conf_mn*100:.2f}%"
                )

                st.progress(conf_mn)

                st.bar_chart(
                    {
                        CLASS_NAMES[i]: float(pred_mn[i])
                        for i in range(len(CLASS_NAMES))
                    }
                )

                st.markdown("</div>", unsafe_allow_html=True)

            # ============================================
            # EFFICIENTNETB0
            # ============================================
            with col2:

                st.markdown("""
                <div style="
                background: rgba(255,255,255,0.04);
                padding:20px;
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.06);
                ">
                """, unsafe_allow_html=True)

                st.markdown("### EfficientNetB0")

                st.success(
                    f"Prediksi: {CLASS_NAMES[idx_ef]}"
                )

                st.write(
                    f"Confidence: {conf_ef*100:.2f}%"
                )

                st.progress(conf_ef)

                st.bar_chart(
                    {
                        CLASS_NAMES[i]: float(pred_ef[i])
                        for i in range(len(CLASS_NAMES))
                    }
                )

                st.markdown("</div>", unsafe_allow_html=True)

            # ============================================
            # BEST MODEL
            # ============================================
            better_model = (
                "EfficientNetB0"
                if conf_ef > conf_mn
                else "MobileNetV2"
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.success(
                f"🏆 Model dengan confidence tertinggi: {better_model}"
            )

            # ============================================
            # TOP 3 PREDICTION
            # ============================================
            st.subheader("🥇 Top 3 Prediksi")

            top3_idx = np.argsort(
                pred_ef
            )[-3:][::-1]

            for i in top3_idx:

                st.write(
                    f"{CLASS_NAMES[i]} "
                    f"({pred_ef[i]*100:.2f}%)"
                )

    except Exception as e:

        st.exception(e)

# ============================================
# FOOTER
# ============================================
st.markdown("---")

st.caption("""
Skripsi:
Perbandingan Efektivitas dan Efisiensi 
Model Transfer Learning MobileNetV2 
dan EfficientNetB0 pada Klasifikasi 
Penyakit Daun Tomat
""")
# ============================================
# CLASS NAMES
# ============================================
CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy"
]

# ============================================
# CONFIDENCE THRESHOLD
# ============================================
CONF_THRESHOLDS = {
    "FF": 0.50,
    "FT10": 0.53,
    "FT20": 0.51,
    "FT30": 0.50
}

# ============================================
# MODEL CONFIG
# ============================================
MODEL_URLS = {

    "FF": {

        "MobileNetV2": {
            "file_id": "1K2Cviwt7P5-HLjwpgoX3QhKsSHP-n3RC",
            "path": "mobilenetv2_ff.keras"
        },

        "EfficientNetB0": {
            "file_id": "1h09odXiKLJogczI_bTYxzNWIvLXKBR4w",
            "path": "efficientnetb0_ff.keras"
        }
    },

    "FT10": {

        "MobileNetV2": {
            "file_id": "1dtZ7pVeOlXzS-xeu_H_SUBXOdg6gjogX",
            "path": "mobilenetv2_ft10.keras"
        },

        "EfficientNetB0": {
            "file_id": "1gVyb03l2FcAwV2AP5NtAG-TFfvcETwiD",
            "path": "efficientnetb0_ft10.keras"
        }
    },

    "FT20": {

        "MobileNetV2": {
            "file_id": "1-chX-5cUA7KlcvDKtYRzfRi6D37qW_XE",
            "path": "mobilenetv2_ft20.keras"
        },

        "EfficientNetB0": {
            "file_id": "1HS9ZRwFlPVXyGNUOjddYWwabtNfNkOH0",
            "path": "efficientnetb0_ft20.keras"
        }
    },

    "FT30": {

        "MobileNetV2": {
            "file_id": "1kcSYiL0BhoqHFSn9kONmowetPwWVePW5",
            "path": "mobilenetv2_ft30.keras"
        },

        "EfficientNetB0": {
            "file_id": "19COA5SOO3eA9ox50mhgT1M_ho-cwiQ7y",
            "path": "efficientnetb0_ft30.keras"
        }
    }
}

# ============================================
# DOWNLOAD MODEL
# ============================================
def download_model(file_id, output_path):

    try:

        if not os.path.exists(output_path):

            url = f"https://drive.google.com/uc?id={file_id}"

            with st.spinner(f"Downloading {output_path} ..."):

                gdown.download(
                    url,
                    output_path,
                    quiet=False
                )

            if not os.path.exists(output_path):

                raise Exception(
                    f"Failed to download {output_path}"
                )

    except Exception as e:

        st.error(f"Download error: {str(e)}")
        raise e

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_single_model(model_name, variant):

    try:

        info = MODEL_URLS[variant][model_name]

        # DOWNLOAD MODEL
        download_model(
            info["file_id"],
            info["path"]
        )

        # VALIDASI FILE
        if not os.path.exists(info["path"]):

            raise Exception(
                f"Model file not found: {info['path']}"
            )

        # LOAD MODEL
        model = tf.keras.models.load_model(
            info["path"],
            compile=False
        )

        return model

    except Exception as e:

        st.error(f"Model load error: {str(e)}")
        raise e

# ============================================
# PREPROCESSING
# ============================================
def preprocess_mobilenet(img: Image.Image):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    img = mobilenet_preprocess(img)

    return img


def preprocess_efficientnet(img: Image.Image):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    img = efficientnet_preprocess(img)

    return img

# ============================================
# UI
# ============================================
st.title("Klasifikasi Penyakit Daun Tomat")

st.write(
    """
    Sistem ini membandingkan performa klasifikasi penyakit daun tomat 
    menggunakan model transfer learning MobileNetV2 dan EfficientNetB0 
    pada berbagai skenario pelatihan.
    """
)

st.markdown("---")

# ============================================
# INFORMASI PENYAKIT
# ============================================
st.subheader("Informasi Kelas Penyakit")

with st.expander("🟤 Bacterial Spot"):
    st.write(
        "Penyakit bercak bakteri yang menyebabkan "
        "bercak kecil coklat gelap pada daun."
    )

with st.expander("🟠 Early Blight"):
    st.write(
        "Ditandai bercak melingkar seperti cincin target pada daun."
    )

with st.expander("⚫ Late Blight"):
    st.write(
        "Penyakit dengan bercak gelap basah yang cepat menyebar."
    )

with st.expander("🟢 Leaf Mold"):
    st.write(
        "Menyebabkan bercak kekuningan dan pertumbuhan jamur."
    )

with st.expander("🟡 Septoria Leaf Spot"):
    st.write(
        "Bercak kecil abu-abu dengan tepi gelap."
    )

with st.expander("🕷 Spider Mites"):
    st.write(
        "Serangan tungau yang merusak jaringan daun."
    )

with st.expander("🎯 Target Spot"):
    st.write(
        "Bercak melingkar menyerupai target."
    )

with st.expander("🦠 Tomato Mosaic Virus"):
    st.write(
        "Virus yang menyebabkan pola mosaik."
    )

with st.expander("🌿 Tomato Yellow Leaf Curl Virus"):
    st.write(
        "Virus yang menyebabkan daun menguning dan melengkung."
    )

with st.expander("✅ Healthy"):
    st.write(
        "Daun tomat sehat tanpa gejala penyakit."
    )

# ============================================
# MODEL SELECTOR
# ============================================
variant = st.selectbox(
    "Pilih Skenario Model",
    ["FF", "FT10", "FT20", "FT30"]
)

# ============================================
# FILE UPLOADER
# ============================================
uploaded_file = st.file_uploader(
    "Upload Gambar Daun Tomat",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# INFERENCE
# ============================================
if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.subheader("Gambar Input")

        col_l, col_c, col_r = st.columns([1, 2, 1])

        with col_c:

            st.image(
                image,
                use_container_width=True
            )

        st.markdown("---")

        # ============================================
        # PREPROCESS
        # ============================================
        x_mn = preprocess_mobilenet(image)
        x_ef = preprocess_efficientnet(image)

        # ============================================
        # LOAD MODEL
        # ============================================
        with st.spinner("Loading MobileNetV2 ..."):

            model_mn = load_single_model(
                "MobileNetV2",
                variant
            )

        with st.spinner("Loading EfficientNetB0 ..."):

            model_ef = load_single_model(
                "EfficientNetB0",
                variant
            )

        # ============================================
        # PREDICTION
        # ============================================
        pred_mn = model_mn.predict(
            x_mn,
            verbose=0
        )[0]

        pred_ef = model_ef.predict(
            x_ef,
            verbose=0
        )[0]

        # ============================================
        # CONFIDENCE
        # ============================================
        conf_mn = float(np.max(pred_mn))
        conf_ef = float(np.max(pred_ef))

        idx_mn = int(np.argmax(pred_mn))
        idx_ef = int(np.argmax(pred_ef))

        threshold = CONF_THRESHOLDS[variant]

        # ============================================
        # CONFIDENCE GATE
        # ============================================
        if conf_mn < threshold or conf_ef < threshold:

            st.markdown("---")

            st.markdown(
                """
                <h4 style='text-align:center; color:red;'>
                Silakan upload ulang gambar daun tomat
                </h4>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Gambar mungkin bukan daun tomat "
                "atau kualitas gambar kurang baik."
            )

        else:

            st.markdown("---")

            st.subheader(
                f"Hasil Prediksi ({variant})"
            )

            col1, col2 = st.columns(2)

            # ============================================
            # MOBILENETV2
            # ============================================
            with col1:

                st.markdown("### MobileNetV2")

                st.write(
                    f"**Prediksi:** "
                    f"{CLASS_NAMES[idx_mn]}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{conf_mn*100:.2f}%"
                )

                st.bar_chart(
                    {
                        CLASS_NAMES[i]: float(pred_mn[i])
                        for i in range(len(CLASS_NAMES))
                    }
                )

            # ============================================
            # EFFICIENTNETB0
            # ============================================
            with col2:

                st.markdown("### EfficientNetB0")

                st.write(
                    f"**Prediksi:** "
                    f"{CLASS_NAMES[idx_ef]}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{conf_ef*100:.2f}%"
                )

                st.bar_chart(
                    {
                        CLASS_NAMES[i]: float(pred_ef[i])
                        for i in range(len(CLASS_NAMES))
                    }
                )

    except Exception as e:

        st.exception(e)

# ============================================
# FOOTER
# ============================================
st.markdown("---")

st.caption(
    """
    Skripsi:
    Perbandingan Efektivitas dan Efisiensi 
    Model Transfer Learning MobileNetV2 
    dan EfficientNetB0 pada Klasifikasi 
    Penyakit Daun Tomat
    """
)
