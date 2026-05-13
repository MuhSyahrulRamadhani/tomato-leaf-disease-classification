# =====================================================
# IMPORT LIBRARY
# =====================================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import gdown

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess
)

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as efficientnet_preprocess
)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Tomato Disease Detection",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODERN UI
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */
.stApp {
    background:
    radial-gradient(circle at top left, #16325B 0%, transparent 25%),
    radial-gradient(circle at bottom right, #14532d 0%, transparent 25%),
    linear-gradient(135deg, #020617 0%, #07111f 50%, #020617 100%);
    color: white;
}

/* =====================================================
CONTAINER
===================================================== */
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* =====================================================
SIDEBAR
===================================================== */
[data-testid="stSidebar"] {
    background: rgba(10,15,25,0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* =====================================================
TITLE
===================================================== */
.main-title {
    font-size: 58px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 10px;
}

.gradient-text {
    background: linear-gradient(
        90deg,
        #4ade80,
        #22d3ee
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #9ca3af;
    font-size: 18px;
    margin-top: 10px;
}

/* =====================================================
GLASS CARD
===================================================== */
.glass-card {

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 25px;

    margin-bottom: 20px;

    box-shadow: 0 0 25px rgba(0,0,0,0.2);
}

/* =====================================================
METRIC CARD
===================================================== */
.metric-card {

    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.07),
        rgba(255,255,255,0.03)
    );

    border-radius: 22px;

    border: 1px solid rgba(255,255,255,0.08);

    padding: 24px;

    text-align: center;

    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-4px);
}

.metric-title {
    color: #9ca3af;
    font-size: 14px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white;
}

/* =====================================================
UPLOAD BOX
===================================================== */
[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.05);

    border: 2px dashed rgba(255,255,255,0.15);

    border-radius: 20px;

    padding: 20px;
}

/* =====================================================
SELECTBOX
===================================================== */
.stSelectbox label {
    color: white !important;
    font-weight: 700;
}

.stSelectbox div[data-baseweb="select"] > div {

    background: rgba(255,255,255,0.06);

    border-radius: 14px;

    border: 1px solid rgba(255,255,255,0.08);

    color: white;
}

/* =====================================================
BUTTON
===================================================== */
.stButton button {

    width: 100%;

    background: linear-gradient(
        90deg,
        #22c55e,
        #06b6d4
    );

    color: white;

    border: none;

    border-radius: 16px;

    padding: 14px;

    font-weight: 700;
}

/* =====================================================
PREDICTION CARD
===================================================== */
.pred-card {

    background: linear-gradient(
        135deg,
        rgba(34,197,94,0.15),
        rgba(6,182,212,0.08)
    );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    margin-bottom: 20px;
}

/* =====================================================
TEXT
===================================================== */
h1,h2,h3,h4,h5,h6 {
    color: white !important;
}

p,span,label,div {
    color: #d1d5db;
}

/* =====================================================
FOOTER
===================================================== */
.footer {
    text-align:center;
    color:#6b7280;
    margin-top:50px;
    padding:20px;
}

/* =====================================================
MOBILE
===================================================== */
@media(max-width:768px){

    .main-title{
        font-size:38px;
    }

    .subtitle{
        font-size:15px;
    }

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# CLASS NAMES
# =====================================================
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

# =====================================================
# CONFIDENCE THRESHOLD
# =====================================================
CONF_THRESHOLDS = {
    "FF": 0.50,
    "FT10": 0.53,
    "FT20": 0.51,
    "FT30": 0.50
}

# =====================================================
# MODEL CONFIG
# =====================================================
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

# =====================================================
# DOWNLOAD MODEL
# =====================================================
def download_model(file_id, output_path):

    if not os.path.exists(output_path):

        url = f"https://drive.google.com/uc?id={file_id}"

        with st.spinner(f"Downloading {output_path} ..."):

            gdown.download(
                url,
                output_path,
                quiet=False
            )

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model(model_name, variant):

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

# =====================================================
# PREPROCESS
# =====================================================
def preprocess_mobilenet(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    return mobilenet_preprocess(img)

def preprocess_efficientnet(img):

    img = img.convert("RGB")
    img = img.resize((224,224))

    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    return efficientnet_preprocess(img)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.markdown("# 🍅 Tomato AI")

    st.caption(
        "Deep Learning Classification System"
    )

    st.markdown("---")

    st.markdown("### ⚙ Pilih Skenario Model")

    variant = st.selectbox(
        "",
        ["FF", "FT10", "FT20", "FT30"]
    )

    st.markdown("---")

    st.metric(
        "Jumlah Kelas",
        "10"
    )

    st.metric(
        "Input Size",
        "224x224"
    )

    st.metric(
        "Framework",
        "TensorFlow"
    )

    st.markdown("---")

    st.markdown("""
    ### 🧠 Model
    - MobileNetV2
    - EfficientNetB0
    """)

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class='glass-card'>

<div class='main-title'>
🍅 <span class='gradient-text'>
AI Tomato Disease Detection
</span>
</div>

<div class='subtitle'>
Sistem klasifikasi penyakit daun tomat menggunakan
Transfer Learning MobileNetV2 dan EfficientNetB0.
</div>

</div>
""", unsafe_allow_html=True)

# =====================================================
# METRIC CARDS
# =====================================================
col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class='metric-card'>
        <div class='metric-title'>Classes</div>
        <div class='metric-value'>10</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class='metric-card'>
        <div class='metric-title'>Models</div>
        <div class='metric-value'>2</div>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Scenario</div>
        <div class='metric-value'>{variant}</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# UPLOAD SECTION
# =====================================================
st.write("")

st.markdown("""
<div class='glass-card'>
<h3>📤 Upload Gambar Daun Tomat</h3>
<p>
Upload gambar daun tomat untuk dilakukan prediksi penyakit menggunakan AI.
</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# INFERENCE
# =====================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    # =================================================
    # IMAGE PREVIEW
    # =================================================
    with col1:

        st.markdown("""
        <div class='glass-card'>
        <h3>🖼 Preview Gambar</h3>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

    # =================================================
    # PREPROCESS
    # =================================================
    x_mn = preprocess_mobilenet(image)
    x_ef = preprocess_efficientnet(image)

    # =================================================
    # LOAD MODEL
    # =================================================
    with st.spinner("Loading AI Models..."):

        model_mn = load_model(
            "MobileNetV2",
            variant
        )

        model_ef = load_model(
            "EfficientNetB0",
            variant
        )

    # =================================================
    # PREDICTION
    # =================================================
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

    threshold = CONF_THRESHOLDS[variant]

    # =================================================
    # RESULT
    # =================================================
    with col2:

        st.markdown("""
        <div class='glass-card'>
        <h3>🤖 Hasil Prediksi AI</h3>
        </div>
        """, unsafe_allow_html=True)

        if conf_mn < threshold or conf_ef < threshold:

            st.error(
                "Silakan upload ulang gambar daun tomat."
            )

            st.caption(
                "Gambar mungkin bukan daun tomat atau kualitas kurang baik."
            )

        else:

            # =============================================
            # MOBILE NET
            # =============================================
            st.markdown(f"""
            <div class='pred-card'>

            <h3>📱 MobileNetV2</h3>

            <h2 style='color:#4ade80;'>
            {CLASS_NAMES[idx_mn]}
            </h2>

            </div>
            """, unsafe_allow_html=True)

            st.progress(conf_mn)

            st.write(
                f"Confidence: {conf_mn*100:.2f}%"
            )

            # =============================================
            # EFFICIENT NET
            # =============================================
            st.markdown(f"""
            <div class='pred-card'>

            <h3>⚡ EfficientNetB0</h3>

            <h2 style='color:#22d3ee;'>
            {CLASS_NAMES[idx_ef]}
            </h2>

            </div>
            """, unsafe_allow_html=True)

            st.progress(conf_ef)

            st.write(
                f"Confidence: {conf_ef*100:.2f}%"
            )

    # =================================================
    # CHART
    # =================================================
    st.write("")

    st.markdown("""
    <div class='glass-card'>
    <h3>📈 Probability Distribution</h3>
    </div>
    """, unsafe_allow_html=True)

    chart_data = pd.DataFrame({

        "Class": CLASS_NAMES,

        "MobileNetV2": pred_mn,

        "EfficientNetB0": pred_ef
    })

    st.bar_chart(
        chart_data.set_index("Class")
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class='footer'>

Skripsi —
Perbandingan Efektivitas dan Efisiensi
Model Transfer Learning MobileNetV2
dan EfficientNetB0 pada Klasifikasi
Penyakit Daun Tomat

</div>
""", unsafe_allow_html=True)
