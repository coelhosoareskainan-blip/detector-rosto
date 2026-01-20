from services.face_api import get_embeddings
import streamlit as st
import numpy as np
from PIL import Image
import os
import cv2
import json
from datetime import datetime
from io import BytesIO

# =====================
# CONFIGURAÇÕES GERAIS
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial (IA Real)")

DB_FILE = "data/faces.json"
HISTORY_FILE = "history.json"

LIMIAR = 0.35
CONF_MINIMA = 70.0

os.makedirs("data", exist_ok=True)

# =====================
# JSON
# =====================

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

db = load_json(DB_FILE, {})
history = load_json(HISTORY_FILE, [])

# =====================
# CASCADE
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
        scaleFactor=1.15,
        minNeighbors=7,
        minSize=(100, 100)
    )

def distancia(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def crop_face(img_rgb, box):
    x, y, w, h = box
    face = img_rgb[y:y+h, x:x+w]
    pil = Image.fromarray(face)
    buf = BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return buf

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro de rosto")

nome = st.text_input("Nome da pessoa")
arquivo = st.file_uploader(
    "Envie uma imagem para cadastro",
    ["jpg", "jpeg", "png"],
    key="cadastro"
)

if arquivo and nome:
    img = Image.open(arquivo).convert("RGB")
    img_np = np.array(img)

    faces = detectar_rostos(img_np)

    if len(faces) == 0:
        st.error("❌ Nenhum rosto detectado.")
    else:
        # 👉 CORREÇÃO AQUI
        x, y, w, h = faces[0]
        face_file = crop_face(img_np, (x, y, w, h))
        embeddings = get_embeddings(face_file)

        if not embeddings:
            st.error("❌ IA não conseguiu extrair embedding.")
        else:
            db[nome] = embeddings[0]
            save_json(DB_FILE, db)

            for (x, y, w, h) in faces:
                cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            st.success(f"✅ Rosto de '{nome}' cadastrado com sucesso!")
            st.image(img_np, use_column_width=True)

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento facial")

arquivo2 = st.file_uploader(
    "Envie uma imagem para reconhecimento",
    ["jpg", "jpeg", "png"],
    key="reconhecimento"
)

if arquivo2:
    if not db:
        st.warning("⚠️ Nenhum rosto cadastrado.")
    else:
        img = Image.open(arquivo2).convert("RGB")
        img_np = np.array(img)

        faces = detectar_rostos(img_np)

        if len(faces) == 0:
            st.error("❌ Nenhum rosto detectado.")
        else:
            for (x, y, w, h) in faces:

                face_file = crop_face(img_np, (x, y, w, h))
                embeddings = get_embeddings(face_file)

                if not embeddings:
                    continue

                emb = embeddings[0]

                resultados = []
                for nome_db, emb_db in db.items():
                    d = distancia(emb, emb_db)
                    resultados.append((nome_db, d))

                resultados.sort(key=lambda x: x[1])

                melhor_nome, melhor_dist = resultados[0]
                confianca = max(0, (1 - melhor_dist / LIMIAR)) * 100

                if melhor_dist > LIMIAR or confianca < CONF_MINIMA:
                    label = "DESCONHECIDO"
                    cor = (255, 0, 0)
                    nome_final = "Desconhecido"
                    confianca = 0.0
                else:
                    label = f"{melhor_nome} ({confianca:.1f}%)"
                    cor = (0, 255, 0)
                    nome_final = melhor_nome

                cv2.rectangle(img_np, (x, y), (x+w, y+h), cor, 2)
                cv2.putText(img_np, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

                history.append({
                    "nome": nome_final,
                    "confianca": round(confianca, 1),
                    "distancia": round(melhor_dist, 3),
                    "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                })

            save_json(HISTORY_FILE, history)
            st.image(img_np, use_column_width=True)

# =====================
# DASHBOARD
# =====================

st.divider()
st.header("📊 Dashboard")

if not db:
    st.info("Nenhum rosto cadastrado.")
else:
    st.write(f"Total de rostos cadastrados: {len(db)}")

# =====================
# HISTÓRICO
# =====================

st.divider()
st.header("🕒 Histórico")

for item in reversed(history[-10:]):
    st.write(f"👤 {item['nome']} | {item['confianca']}% | {item['data']}")
