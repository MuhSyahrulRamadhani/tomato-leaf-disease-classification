#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Tomat",
    layout="centered"
)

# =============================
# CLASS NAMES
# SESUAI DATASET SKRIPSI
# =============================
CLASS_NAMES = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Mosaic Virus',
    'Tomato Yellow Leaf Curl Virus',
    'Healthy'
]

# =============================
# CONFIDENCE THRESHOLD
# =============================
CONF_THRESHOLDS = {
    "FF": 0.50,
    "FT10": 0.53,
    "FT20": 0.51,
    "FT30": 0.50
}

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_models():
    models = {
        "FF": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ff.keras",
                compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ff.keras",
                compile=False
            )
        },

        "FT10": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft10.keras",
                compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft10.keras",
                compile=False
            )
        },

        "FT20": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft20.keras",
                compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft20.keras",
                compile=False
            )
        },

        "FT30": {
            "MobileNetV2": tf.keras.models.load_model(
                "mobilenetv2_ft30.keras",
                compile=False
            ),
            "EfficientNetB0": tf.keras.models.load_model(
                "efficientnetb0_ft30.keras",
                compile=False
            )
        }
    }

    return models


MODELS = load_models()

# =============================
# PREPROCESSING
# SESUAI SKRIPSI
# =============================
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


# =============================
# UI
# =============================
st.title("Klasifikasi Penyakit Daun Tomat")

st.write(
    """
    Sistem ini membandingkan performa klasifikasi penyakit daun tomat 
    menggunakan model transfer learning MobileNetV2 dan EfficientNetB0 
    pada berbagai skenario pelatihan.
    """
)

st.markdown("---")

# =============================
# INFORMASI PENYAKIT
# =============================
st.subheader("Informasi Kelas Penyakit")

with st.expander("🟤 Bacterial Spot"):
    st.write(
        "Penyakit bercak bakteri yang menyebabkan munculnya "
        "bercak kecil berwarna coklat gelap pada daun."
    )

with st.expander("🟠 Early Blight"):
    st.write(
        "Ditandai bercak melingkar seperti cincin target "
        "dengan warna coklat pada daun."
    )

with st.expander("⚫ Late Blight"):
    st.write(
        "Penyakit dengan bercak gelap basah yang cepat menyebar "
        "pada permukaan daun."
    )

with st.expander("🟢 Leaf Mold"):
    st.write(
        "Menyebabkan bercak kekuningan pada permukaan atas daun "
        "dan jamur pada bagian bawah daun."
    )

with st.expander("🟡 Septoria Leaf Spot"):
    st.write(
        "Ditandai bercak kecil abu-abu dengan tepi gelap "
        "yang menyebar pada daun."
    )

with st.expander("🕷 Spider Mites"):
    st.write(
        "Serangan tungau yang menyebabkan bercak kuning "
        "dan kerusakan jaringan daun."
    )

with st.expander("🎯 Target Spot"):
    st.write(
        "Memiliki bercak melingkar menyerupai target "
        "dengan pola konsentris."
    )

with st.expander("🦠 Tomato Mosaic Virus"):
    st.write(
        "Virus yang menyebabkan pola mosaik "
        "dan perubahan bentuk daun."
    )

with st.expander("🌿 Tomato Yellow Leaf Curl Virus"):
    st.write(
        "Virus yang menyebabkan daun menguning "
        "dan melengkung."
    )

with st.expander("✅ Healthy"):
    st.write(
        "Daun tomat sehat tanpa gejala penyakit."
    )

# =============================
# MODEL SELECTOR
# =============================
variant = st.selectbox(
    "Pilih skenario model",
    ["FF", "FT10", "FT20", "FT30"]
)

# =============================
# FILE UPLOADER
# =============================
uploaded_file = st.file_uploader(
    "Upload gambar daun tomat",
    type=["jpg", "jpeg", "png"]
)

# =============================
# INFERENCE
# =============================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Gambar Input")

    col_l, col_c, col_r = st.columns([1, 2, 1])

    with col_c:
        st.image(image, use_container_width=True)

    # =============================
    # PREPROCESS
    # =============================
    x_mn = preprocess_mobilenet(image)
    x_ef = preprocess_efficientnet(image)

    # =============================
    # LOAD MODEL
    # =============================
    model_mn = MODELS[variant]["MobileNetV2"]
    model_ef = MODELS[variant]["EfficientNetB0"]

    # =============================
    # PREDICT
    # =============================
    pred_mn = model_mn.predict(x_mn, verbose=0)[0]
    pred_ef = model_ef.predict(x_ef, verbose=0)[0]

    # =============================
    # CONFIDENCE
    # =============================
    conf_mn = float(np.max(pred_mn))
    conf_ef = float(np.max(pred_ef))

    idx_mn = int(np.argmax(pred_mn))
    idx_ef = int(np.argmax(pred_ef))

    threshold = CONF_THRESHOLDS[variant]

    # =============================
    # CONFIDENCE GATE
    # =============================
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

        st.subheader(f"Hasil Prediksi ({variant})")

        col1, col2 = st.columns(2)

        # =============================
        # MOBILENETV2
        # =============================
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

        # =============================
        # EFFICIENTNETB0
        # =============================
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

# =============================
# FOOTER
# =============================
st.markdown("---")

st.caption(
    """
    Skripsi:
    Perbandingan Efektivitas dan Efisiensi Model Transfer Learning 
    MobileNetV2 dan EfficientNetB0 pada Klasifikasi Penyakit Daun Tomat
    """
)

