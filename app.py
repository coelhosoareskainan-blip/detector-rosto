import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Detector de Rosto", layout="centered")
st.title("🧠 Detector de Rosto – Upload de Imagem")

DB_FILE = "face_db.npz"

# ======================
# UTILIDADES
# ======================

def load_db():
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        return dict(data)
    return {}

def save_db(db):
    np.savez(DB_FILE, **db)

def detectar_rosto(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    return img_bgr[y:y+h, x:x+w], (x, y, w, h)

def extrair_histograma(rosto):
    hist = cv2.calcHist(
        [rosto], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist

db = load_db()

# ======================
# CADASTRO
# ======================

st.header("🅱️ Cadastro de rosto")

nome = st.text_input("Nome da pessoa")
arquivo = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if arquivo and nome:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    rosto, box = detectar_rosto(img_bgr)

    if rosto is None:
        st.error("❌ Nenhum rosto detectado.")
    else:
        hist = extrair_histograma(rosto)
        db[nome] = hist
        save_db(db)

        x, y, w, h = box
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h)_
