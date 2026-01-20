import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json
import os
from deepface import DeepFace

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial Profissional", layout="centered")
st.title("🧠 Reconhecimento Facial • Nível Profissional")

DB_FILE = "data/faces.json"
os.makedirs("data", exist_ok=True)

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
DIST_THRESHOLD = 0.35

# =====================
# BANCO
# =====================

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

db = load_db()

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro de Pessoa")

nome = st.text_input("Nome da pessoa")
arquivo = st.file_uploader("Imagem para cadastro", ["jpg", "jpeg", "png"])

if arquivo and nome:
    img = np.array(Image.open(arquivo).convert("RGB"))

    try:
        reps = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
            align=True
        )

        db.setdefault(nome, [])

        for r in reps:
            db[nome].append(r["embedding"])

        save_db(db)
        st.success(f"✅ {nome} cadastrado com sucesso")

    except Exception as e:
        st.error("Erro ao cadastrar")
        st.code(str(e))

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento Facial")

arquivo2 = st.file_uploader(
    "Imagem para reconhecimento",
    ["jpg", "jpeg", "png"],
    key="rec"
)

if arquivo2 and db:
    img_pil = Image.open(arquivo2).convert("RGB")
    img = np.array(img_pil)
    draw = ImageDraw.Draw(img_pil)

    try:
        detections = DeepFace.extract_faces(
            img_path=img,
            detector_backend=DETECTOR,
            enforce_detection=False,
            align=True
        )

        for det in detections:
            face_img = det["face"]
            region = det["facial_area"]

            rep = DeepFace.represent(
                img_path=face_img,
                model_name=MODEL_NAME,
                detector_backend="skip"
            )[0]["embedding"]

            melhor_nome = "DESCONHECIDO"
            melhor_dist = 1.0

            for nome_db, embs in db.items():
                for emb_db in embs:
                    cos = np.dot(rep, emb_db) / (
                        np.linalg.norm(rep) * np.linalg.norm(emb_db)
                    )
                    dist = 1 - cos

                    if dist < melhor_dist:
                        melhor_dist = dist
                        melhor_nome = nome_db

            if melhor_dist <= DIST_THRESHOLD:
                label = melhor_nome
                cor = "green"
            else:
                label = "DESCONHECIDO"
                cor = "red"

            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            draw.rectangle([x, y, x + w, y + h], outline=cor, width=3)
            draw.text((x, y - 20 if y > 20 else y + h + 5), label, fill=cor)

        st.image(img_pil, use_column_width=True)

    except Exception as e:
        st.error("Erro no reconhecimento")
        st.code(str(e))

elif arquivo2 and not db:
    st.warning("Nenhuma pessoa cadastrada")
