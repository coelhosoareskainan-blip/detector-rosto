import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial (IA Real)")

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# =====================
# LOAD MODELO (HAAR)
# =====================

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Erro ao carregar haarcascade_frontalface_default.xml")
    st.stop()

# =====================
# FUNÇÃO DETECÇÃO
# =====================

def detectar_rostos(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces

# =====================
# DETECÇÃO VIA IMAGEM
# =====================

st.header("📷 Detectar rosto em imagem")

arquivo = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"]
)

if arquivo:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)

    faces = detectar_rostos(img_np)

    if len(faces) == 0:
        st.error("❌ Nenhum rosto detectado.")
    else:
        st.success(f"✅ {len(faces)} rosto(s) detectado(s)")

        for (x, y, w, h) in faces:
            cv2.rectangle(
                img_np,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        st.image(img_np, caption="Resultado da detecção", use_column_width=True)

# =====================
# WEBCAM
# =====================

st.divider()
st.header("🎥 Webcam")

if IS_CLOUD:
    st.warning("🚫 Webcam desativada no Streamlit Cloud")
else:
    st.info("✅ Rode localmente para usar webcam:")
    st.code("streamlit run app.py")
