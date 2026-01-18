from services.face_api import get_embedding
import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import json
from datetime import datetime

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial (IA Real)")

DB_FILE = "data/faces.json"
HISTORY_FILE = "history.json"

LIMIAR = 0.25          # LIMIAR DURO
CONF_MINIMA = 80.0     # CONFIANÇA MÍNIMA OBRIGATÓRIA

os.makedirs("data", exist_ok=True)

# =====================
# UTILIDADES
# =====================

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

db = load_json(DB_FILE, {})
history = load_json(HISTORY_FILE, [])

# =====================
# CASCADE
# =====================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================
# FUNÇÕES
# =====================

def detectar_rostos(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(80, 80)
    )

def distancia(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro")

nome = st.text_input("Nome")
arquivo = st.file_uploader("Imagem", ["jpg", "png"], key="cad")

if arquivo and nome:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)

    faces = detectar_rostos(img_np)
    if len(faces) == 0:
        st.error("❌ Nenhum rosto")
    else:
        emb = get_embedding(arquivo)
        if emb:
            db[nome] = emb
            save_json(DB_FILE, db)
            st.success(f"✅ {nome} cadastrado")
            st.image(img_np)

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento")

arquivo2 = st.file_uploader("Imagem", ["jpg", "png"], key="rec")

if arquivo2 and db:
    emb = get_embedding(arquivo2)

    if emb is None:
        st.error("❌ IA falhou")
    else:
        resultados = []

        for nome_db, emb_db in db.items():
            d = distancia(emb, emb_db)
            resultados.append((nome_db, d))

        resultados.sort(key=lambda x: x[1])

        melhor_nome, melhor_dist = resultados[0]

        confianca = (1 - melhor_dist / LIMIAR) * 100

        # =====================
        # DECISÃO FINAL
        # =====================

        if melhor_dist > LIMIAR or confianca < CONF_MINIMA:
            melhor_nome = "Desconhecido"
            confianca = 0.0
            st.error("❌ Desconhecido")
        else:
            st.success(f"✅ {melhor_nome} | confiança: {confianca:.1f}%")

        history.append({
            "nome": melhor_nome,
            "confianca": round(confianca, 1),
            "dist": round(melhor_dist, 3),
            "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        })

        save_json(HISTORY_FILE, history)

# =====================
# TESTE IA
# =====================

st.divider()
st.header("🧪 Teste IA")

teste = st.file_uploader("Teste", ["jpg", "png"], key="test")

if teste:
    vec = get_embedding(teste)
    if vec:
        st.success(f"✅ Vetor OK ({len(vec)} números)")
    else:
        st.error("❌ Vetor inválido")
