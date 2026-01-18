from services.face_api import get_embedding
import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import json

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial (IA Real)")

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None
DB_FILE = "data/faces.json"
LIMIAR = 0.35

# =====================
# LOAD CASCADE
# =====================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Erro ao carregar Haar Cascade")
    st.stop()

# =====================
# BANCO DE DADOS
# =====================

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

db = load_db()

# =====================
# DETECÇÃO DE ROSTO
# =====================

def detectar_rostos(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    return faces

# =====================
# DISTÂNCIA
# =====================

def distancia(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro de rosto")

nome = st.text_input("Nome da pessoa")
arquivo = st.file_uploader(
    "Envie uma imagem para cadastro",
    type=["jpg", "jpeg", "png"],
    key="cadastro"
)

if arquivo and nome:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)

    faces = detectar_rostos(img_np)

    if len(faces) == 0:
        st.error("❌ Nenhum rosto detectado.")
    else:
        embedding = get_embedding(arquivo)

        if embedding is None:
            st.error("❌ IA não conseguiu extrair o rosto.")
        else:
            db[nome] = embedding
            save_db(db)
            st.success(f"✅ Rosto de '{nome}' cadastrado com sucesso!")

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento facial")

arquivo2 = st.file_uploader(
    "Envie uma imagem para reconhecimento",
    type=["jpg", "jpeg", "png"],
    key="reconhecimento"
)

if arquivo2:
    img = Image.open(arquivo2).convert("RGB")
    img_np = np.array(img)

    faces = detectar_rostos(img_np)

    if len(faces) == 0:
        st.error("❌ Nenhum rosto detectado.")
    elif not db:
        st.warning("⚠️ Nenhum rosto cadastrado.")
    else:
        emb = get_embedding(arquivo2)

        if emb is None:
            st.error("❌ IA não conseguiu extrair o rosto.")
        else:
            melhor_nome = "Desconhecido"
            melhor_dist = 1.0

            for nome_db, emb_db in db.items():
                d = distancia(emb, emb_db)
                if d < melhor_dist:
                    melhor_dist = d
                    melhor_nome = nome_db

            if melhor_dist > LIMIAR:
                melhor_nome = "Desconhecido"

            confianca = max(0, (1 - melhor_dist / LIMIAR)) * 100

if melhor_nome == "Desconhecido":
    st.error(f"❌ Desconhecido | confiança: {confianca:.1f}%")
else:
    st.success(f"✅ {melhor_nome} | confiança: {confianca:.1f}%")

# =====================
# DASHBOARD
# =====================

st.divider()
st.header("📊 Dashboard")

if not db:
    st.info("Nenhum rosto cadastrado ainda.")
else:
    st.write(f"Total de rostos cadastrados: {len(db)}")

    for nome in list(db.keys()):
        col1, col2 = st.columns([4, 1])
        col1.write(nome)

        if col2.button("❌", key=f"del_{nome}"):
            del db[nome]
            save_db(db)
            st.rerun()

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
