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
    page_title="Tomato AI Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background: linear-gradient(
        135deg,
        #071b17,
        #0b2d26,
        #123c32
    );
    color: white;
}

/* RESPONSIVE CONTAINER */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #050505;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* SELECTBOX LABEL */
.stSelectbox label {
    color: white !important;
    font-weight: bold;
}

/* SELECTBOX INPUT */
.stSelectbox div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
    border-radius: 10px;
}

/* DROPDOWN MENU */
div[data-baseweb="popover"] * {
    color: black !important;
}

/* CARD */
.custom-card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}

/* PREDICTION CARD */
.pred-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    margin-bottom: 15px;
}

/* RESULT TITLE */
.result-title {
    font-size: 22px;
    font-weight: bold;
    color: #9effb0;
}

/* CONFIDENCE */
.confidence {
    font-size: 30px;
    font-weight: bold;
    color: #ffe082;
}

/* SMALL TEXT */
.small-text {
    color: #d7d7d7;
    font-size: 14px;
}

/* IMAGE */
img {
    border-radius: 18px;
}

/* MOBILE RESPONSIVE */
@media (max-width: 768px) {

    .result-title {
        font-size: 18px;
    }

    .confidence {
        font-size: 24px;
    }

    .custom-card {
        padding: 15px;
    }

    .pred-card {
        padding: 15px;
    }

    h1 {
        font-size: 28px !important;
    }

    h2 {
        font-size: 22px !important;
    }

    h3 {
        font-size: 18px !important;
    }

}

</style>
""", unsafe_allow_html=True)

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
# DISEASE INFORMATION
# ============================================
DISEASE_INFO = {

    "Bacterial Spot": {
        "cause": "Infeksi bakteri Xanthomonas.",
        "symptom": "Bercak coklat kecil pada daun.",
        "solution": "Gunakan benih sehat dan fungisida."
    },

    "Early Blight": {
        "cause": "Jamur Alternaria solani.",
        "symptom": "Bercak melingkar seperti target.",
        "solution": "Buang daun terinfeksi."
    },

    "Late Blight": {
        "cause": "Jamur Phytophthora infestans.",
        "symptom": "Bercak hitam basah menyebar cepat.",
        "solution": "Kurangi kelembaban dan gunakan fungisida."
    },

    "Leaf Mold": {
        "cause": "Jamur Passalora fulva.",
        "symptom": "Daun menguning dan berjamur.",
        "solution": "Perbaiki sirkulasi udara."
    },

    "Septoria Leaf Spot": {
        "cause": "Jamur Septoria lycopersici.",
        "symptom": "Bercak abu-abu kecil.",
        "solution": "Buang daun yang terinfeksi."
    },

    "Spider Mites": {
        "cause": "Serangan tungau.",
        "symptom": "Daun berbintik kuning.",
        "solution": "Gunakan pestisida tungau."
    },

    "Target Spot": {
        "cause": "Jamur Corynespora cassiicola.",
        "symptom": "Bercak menyerupai target.",
        "solution": "Gunakan fungisida."
    },

    "Tomato Mosaic Virus": {
        "cause": "Virus mosaik tomat.",
        "symptom": "Pola mosaik pada daun.",
        "solution": "Cabut tanaman terinfeksi."
    },

    "Tomato Yellow Leaf Curl Virus": {
        "cause": "Virus TYLCV.",
        "symptom": "Daun menguning dan melengkung.",
        "solution": "Kendalikan kutu putih."
    },

    "Healthy": {
        "cause": "Tanaman sehat.",
        "symptom": "Tidak ada gejala penyakit.",
        "solution": "Pertahankan perawatan tanaman."
    }
}

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
# SIDEBAR
# ============================================
with st.sidebar:

    st.markdown("# 🌿 Tomato AI")

    st.markdown("---")

    selected_variant = st.selectbox(
        "Pilih Skenario Model",
        ["FF", "FT10", "FT20", "FT30"]
    )

    st.markdown("---")

    st.markdown("""
    ### 📌 Tentang Sistem
    
    Sistem AI berbasis Deep Learning untuk:
    
    - Klasifikasi penyakit daun tomat
    - Perbandingan model CNN
    - Evaluasi transfer learning
    - Analisis hasil prediksi
    """)

    st.markdown("---")

    st.markdown("""
    ### 📸 Tips Upload
    
    - Gunakan gambar jelas
    - Fokus pada daun
    - Hindari blur
    - Gunakan cahaya cukup
    """)

    st.markdown("---")

    st.markdown("""
    ### 👨‍🎓 Penelitian Skripsi
    
    Perbandingan Efektivitas dan Efisiensi 
    MobileNetV2 dan EfficientNetB0 
    pada klasifikasi penyakit daun tomat.
    """)

# ============================================
# DOWNLOAD MODEL
# ============================================
def download_model(file_id, output_path):

    if not os.path.exists(output_path):

        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner(f"Downloading {output_path} ..."):

            gdown.download(
                url,
                output_path,
                quiet=False
            )

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_single_model(model_name, variant):

    info = MODEL_URLS[variant][model_name]

    download_model(
        info["file_id"],
        info["path"]
    )

    model = tf.keras.models.load_model(
        info["path"],
        compile=False
    )

    return model

# ============================================
# PREPROCESSING
# ============================================
def preprocess_mobilenet(img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    return mobilenet_preprocess(img)

def preprocess_efficientnet(img):

    img = img.convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    return efficientnet_preprocess(img)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="custom-card">

<h1>
🌿 Tomato Leaf Disease Detection
</h1>

<p class="small-text">
Sistem klasifikasi penyakit daun tomat berbasis Deep Learning 
menggunakan MobileNetV2 dan EfficientNetB0.
</p>

</div>
""", unsafe_allow_html=True)

# ============================================
# INFO CARDS
# ============================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="custom-card">
    <h3>📊 Dataset</h3>
    <p>10 kelas penyakit daun tomat.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="custom-card">
    <h3>🧠 Model</h3>
    <p>Skenario: {selected_variant}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="custom-card">
    <h3>⚡ Transfer Learning</h3>
    <p>MobileNetV2 & EfficientNetB0.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# UPLOADER
# ============================================
st.markdown("""
<div class="custom-card">

<h3>📤 Upload Gambar Daun Tomat</h3>

<p class="small-text">
Upload gambar daun tomat untuk dilakukan klasifikasi AI.
</p>

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# INFERENCE
# ============================================
if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        st.markdown("""
        <div class="custom-card">
        <h3>🖼️ Gambar Input</h3>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

        x_mn = preprocess_mobilenet(image)
        x_ef = preprocess_efficientnet(image)

        with st.spinner("Loading AI Models ..."):

            model_mn = load_single_model(
                "MobileNetV2",
                selected_variant
            )

            model_ef = load_single_model(
                "EfficientNetB0",
                selected_variant
            )

        pred_mn = model_mn.predict(
            x_mn,
            verbose=0
        )[0]

        pred_ef = model_ef.predict(
            x_ef,
            verbose=0
        )[0]

        conf_mn = float(np.max(pred_mn))
        conf_ef = float(np.max(pred_ef))

        idx_mn = int(np.argmax(pred_mn))
        idx_ef = int(np.argmax(pred_ef))

        threshold = CONF_THRESHOLDS[selected_variant]

        if conf_mn < threshold or conf_ef < threshold:

            st.error(
                "Gambar tidak valid atau kualitas terlalu rendah."
            )

        else:

            st.markdown("---")

            st.markdown("""
            <div class="custom-card">
            <h2>📊 Hasil Prediksi AI</h2>
            </div>
            """, unsafe_allow_html=True)

            colA, colB = st.columns(2)

            with colA:

                st.markdown(f"""
                <div class="pred-card">

                <h2>MobileNetV2</h2>

                <p class="result-title">
                {CLASS_NAMES[idx_mn]}
                </p>

                <p class="confidence">
                {conf_mn*100:.2f}%
                </p>

                </div>
                """, unsafe_allow_html=True)

                st.bar_chart({
                    CLASS_NAMES[i]: float(pred_mn[i])
                    for i in range(len(CLASS_NAMES))
                })

            with colB:

                st.markdown(f"""
                <div class="pred-card">

                <h2>EfficientNetB0</h2>

                <p class="result-title">
                {CLASS_NAMES[idx_ef]}
                </p>

                <p class="confidence">
                {conf_ef*100:.2f}%
                </p>

                </div>
                """, unsafe_allow_html=True)

                st.bar_chart({
                    CLASS_NAMES[i]: float(pred_ef[i])
                    for i in range(len(CLASS_NAMES))
                })

            # ============================================
            # COMPARISON
            # ============================================
            better_model = (
                "EfficientNetB0"
                if conf_ef > conf_mn
                else "MobileNetV2"
            )

            st.markdown(f"""
            <div class="custom-card">

            <h2>📈 Ringkasan Perbandingan</h2>

            <ul>
            <li>
            MobileNetV2 Confidence:
            <b>{conf_mn*100:.2f}%</b>
            </li>

            <li>
            EfficientNetB0 Confidence:
            <b>{conf_ef*100:.2f}%</b>
            </li>

            <li>
            Model dengan confidence tertinggi:
            <b>{better_model}</b>
            </li>

            </ul>

            </div>
            """, unsafe_allow_html=True)

            # ============================================
            # DISEASE INFO
            # ============================================
            disease_name = CLASS_NAMES[idx_ef]

            if disease_name in DISEASE_INFO:

                info = DISEASE_INFO[disease_name]

                st.markdown(f"""
                <div class="custom-card">

                <h2>🦠 Informasi Penyakit</h2>

                <h3>{disease_name}</h3>

                <p>
                <b>Penyebab:</b><br>
                {info['cause']}
                </p>

                <p>
                <b>Gejala:</b><br>
                {info['symptom']}
                </p>

                <p>
                <b>Penanganan:</b><br>
                {info['solution']}
                </p>

                </div>
                """, unsafe_allow_html=True)

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
