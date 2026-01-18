import streamlit as st
import numpy as np
from PIL import Image
import os

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial (IA Real)")

DB_FILE = "face_db.npz"
LIMIAR = 0.6

IS_CLOUD = os.getenv("STREAMLIT_CLOUD") is not None

# controle de exclusão segura
if "delete_name" not in st.session_state:
    st.session_state.delete_name = None

# =====================
# BANCO DE DADOS
# =====================

def load_db():
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        return dict(data)
    return {}

def save_db(db):
    np.savez(DB_FILE, **db)

def deletar_rosto(nome):
    if nome in db:
        del db[nome]
        save_db(db)

db = load_db()

# =====================
# FUNÇÕES IA
# =====================

def detectar_e_extrair(img_rgb):
    locais = face_recognition.face_locations(img_rgb)
    embeddings = face_recognition.face_encodings(img_rgb, locais)
    return embeddings, locais

def distancia(a, b):
    return np.linalg.norm(a - b)

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro de rosto")

nome = st.text_input("Nome da pessoa")
arquivo = st.file_uploader(
    "Envie uma imagem para cadastro",
    type=["jpg", "jpeg", "png"]
)

if arquivo and nome:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)

    embeddings, _ = detectar_e_extrair(img_np)

    if not embeddings:
        st.error("❌ Nenhum rosto detectado.")
    else:
        db[nome] = embeddings[0]
        save_db(db)
        st.success(f"✅ Rosto de '{nome}' cadastrado com sucesso")

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento facial")

arquivo2 = st.file_uploader(
    "Envie uma imagem para reconhecimento",
    type=["jpg", "jpeg", "png"],
    key="rec"
)

if arquivo2:
    img = Image.open(arquivo2).convert("RGB")
    img_np = np.array(img)

    embeddings, _ = detectar_e_extrair(img_np)

    if not embeddings:
        st.error("❌ Nenhum rosto detectado.")
    elif not db:
        st.warning("⚠️ Nenhum rosto cadastrado.")
    else:
        for emb in embeddings:
            melhor_nome = "Desconhecido"
            melhor_dist = 1.0

            for nome_db, emb_db in db.items():
                d = distancia(emb, emb_db)
                if d < melhor_dist:
                    melhor_dist = d
                    melhor_nome = nome_db

            if melhor_dist > LIMIAR:
                melhor_nome = "Desconhecido"

            st.success(f"👤 {melhor_nome} | distância: {melhor_dist:.3f}")

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
            st.session_state.delete_name = nome

# exclusão segura (FORA do loop)
if st.session_state.delete_name:
    deletar_rosto(st.session_state.delete_name)
    st.session_state.delete_name = None
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
