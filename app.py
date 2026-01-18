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

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

DB_FILE = "data/faces.json"
HISTORY_FILE = "history.json"
LIMIAR = 0.35

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
# LOAD CASCADE
# =====================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    st.error("❌ Erro ao carregar Haar Cascade")
    st.stop()

# =====================
# FUNÇÕES
# =====================

def detectar_rostos(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

def distancia(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

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
            save_json(DB_FILE, db)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

            st.success(f"✅ Rosto de '{nome}' cadastrado com sucesso!")
            st.image(img_np, caption=f"Rosto cadastrado: {nome}", use_column_width=True)

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
# =====================
# MATCH PROFISSIONAL
# =====================

resultados = []

for nome_db, emb_db in db.items():
    d = distancia(emb, emb_db)
    resultados.append((nome_db, d))

# ordenar por distância
resultados.sort(key=lambda x: x[1])

melhor_nome, melhor_dist = resultados[0]

# segurança extra: segundo mais próximo
segundo_dist = resultados[1][1] if len(resultados) > 1 else 1.0

# ========= DECISÃO FINAL =========
DESCONHECIDO = False

# regra 1: distância absoluta
if melhor_dist > LIMIAR:
    DESCONHECIDO = True

# regra 2: distância muito próxima do segundo
if abs(segundo_dist - melhor_dist) < 0.05:
    DESCONHECIDO = True

# ========= RESULTADO =========
if DESCONHECIDO:
    st.error("❌ Desconhecido")
    melhor_nome = "Desconhecido"
    confianca = 0.0
else:
    confianca = (1 - melhor_dist / LIMIAR) * 100
    st.success(f"✅ {melhor_nome} | confiança: {confianca:.1f}%")


            st.image(img_np, caption="Resultado do reconhecimento", use_column_width=True)

            history.append({
                "nome": melhor_nome,
                "confianca": round(confianca, 1),
                "distancia": round(melhor_dist, 3),
                "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            })

            save_json(HISTORY_FILE, history)

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
            save_json(DB_FILE, db)
            st.rerun()

# =====================
# HISTÓRICO
# =====================

st.divider()
st.header("🕒 Histórico de Reconhecimentos")

if not history:
    st.info("Nenhum reconhecimento ainda.")
else:
    for item in reversed(history[-10:]):
        st.write(f"👤 {item['nome']} | {item['confianca']}% | {item['data']}")

# =====================
# TESTE DA IA
# =====================

st.divider()
st.header("🧪 Teste da IA")

teste_img = st.file_uploader(
    "Envie uma imagem para TESTE DA IA",
    type=["jpg", "jpeg", "png"],
    key="teste_ia"
)

if teste_img:
    vec = get_embedding(teste_img)

    if vec is None:
        st.error("❌ IA NÃO RETORNOU VETOR")
    else:
        st.success(f"✅ IA OK | vetor com {len(vec)} números")
