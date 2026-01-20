from deepface import DeepFace
import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import cv2

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial Profissional", layout="centered")
st.title("🧠 Reconhecimento Facial • Nível Profissional")

DB_FILE = "data/faces.json"
os.makedirs("data", exist_ok=True)

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
DIST_THRESHOLD = 0.35  # 🔥 nível policial

# =====================
# DB
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
            enforce_detection=True
        )

        if nome not in db:
            db[nome] = []

        for r in reps:
            db[nome].append(r["embedding"])

        save_db(db)
        st.success(f"✅ {nome} cadastrado ({len(reps)} rosto(s))")

    except Exception as e:
        st.error("❌ Erro no cadastro")
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
    img = np.array(Image.open(arquivo2).convert("RGB"))

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
                    d = np.dot(rep, emb_db) / (
                        np.linalg.norm(rep) * np.linalg.norm(emb_db)
                    )
                    dist = 1 - d  # cosine distance

                    if dist < melhor_dist:
                        melhor_dist = dist
                        melhor_nome = nome_db

            if melhor_dist <= DIST_THRESHOLD:
                label = f"{melhor_nome}"
                cor = (0, 255, 0)
            else:
                label = "DESCONHECIDO"
                cor = (255, 0, 0)

            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(img, (x, y), (x+w, y+h), cor, 2)
            cv2.putText(
                img,
                label,
                (x, y-10 if y > 20 else y+h+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                cor,
                2
            )

        st.image(img, use_column_width=True)

    except Exception as e:
        st.error("❌ Erro no reconhecimento")
        st.code(str(e))
