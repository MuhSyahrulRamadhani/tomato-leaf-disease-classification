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
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================
# MODERN PROFESSIONAL UI
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
PORTRAIT LAYOUT
===================================================== */
.block-container {

    max-width: 820px;

    padding-top: 1.5rem;

    padding-bottom: 2rem;
}

/* =====================================================
SIDEBAR
===================================================== */
[data-testid="stSidebar"] {

    background: rgba(10,15,25,0.96);

    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* =====================================================
TITLE
===================================================== */
.main-title {

    font-size: 48px;

    font-weight: 800;

    line-height: 1.1;

    margin-bottom: 10px;

    text-align: center;
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

    font-size: 17px;

    margin-top: 10px;

    text-align: center;
}

/* =====================================================
GLASS CARD
===================================================== */
.glass-card {

    background: rgba(255,255,255,0.05);

    backdrop-filter: blur(14px);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 22px;

    margin-bottom: 20px;

    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

/* =====================================================
METRIC CARD
===================================================== */
.metric-card {

    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.06),
        rgba(255,255,255,0.03)
    );

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.08);

    padding: 20px;

    text-align: center;
}

.metric-title {

    color: #9ca3af;

    font-size: 14px;
}

.metric-value {

    font-size: 26px;

    font-weight: 700;

    color: white;
}

/* =====================================================
UPLOAD BOX
===================================================== */
[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.04);

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
PREVIEW IMAGE
===================================================== */
.preview-image {

    display: flex;

    justify-content: center;

    margin-top: 10px;
}

.preview-image img {

    border-radius: 18px;

    border: 1px solid rgba(255,255,255,0.08);

    max-width: 220px !important;

    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}

/* =====================================================
TABLE
===================================================== */
[data-testid="stDataFrame"] {

    border-radius: 18px;

    overflow: hidden;

    border: 1px solid rgba(255,255,255,0.08);
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
        font-size:34px;
    }

    .subtitle{
        font-size:15px;
    }

    .preview-image img{
        max-width:180px !important;
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
# METRIC CARD
# =====================================================
col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class='metric-card'>
        <div class='metric-title'>Classes</div>
        <div class='metric-value'>10</div>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Scenario</div>
        <div class='metric-value'>{variant}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# UPLOAD SECTION
# =====================================================
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
    # LAYOUT IMAGE + RESULT
    # =================================================
    col1, col2 = st.columns([1,1])

    # =================================================
    # IMAGE PREVIEW
    # =================================================
    with col1:

        st.markdown("""
        <div class='glass-card'>
        <h3 style='text-align:center;'>
        🖼 Preview Gambar
        </h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='preview-image'>",
            unsafe_allow_html=True
        )

        st.image(
            image,
            width=220
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # =================================================
    # RESULT
    # =================================================
    with col2:

        st.markdown("""
        <div class='glass-card'>
        <h3 style='text-align:center;'>
        🤖 Hasil Prediksi
        </h3>
        </div>
        """, unsafe_allow_html=True)

        if conf_mn < threshold or conf_ef < threshold:

            st.error(
                "Silakan upload ulang gambar daun tomat."
            )

            st.caption(
                "Gambar kurang sesuai atau kualitas rendah."
            )

        else:

            result_df = pd.DataFrame({

                "Model": [
                    "MobileNetV2",
                    "EfficientNetB0"
                ],

                "Prediksi": [
                    CLASS_NAMES[idx_mn],
                    CLASS_NAMES[idx_ef]
                ],

                "Confidence": [
                    f"{conf_mn*100:.2f}%",
                    f"{conf_ef*100:.2f}%"
                ]
            })

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.write("")

            st.markdown("#### 📱 MobileNetV2")

            st.progress(conf_mn)

            st.markdown("#### ⚡ EfficientNetB0")

            st.progress(conf_ef)

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
        chart_data.set_index("Class"),
        height=350
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
