import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os
import hashlib

st.set_page_config(page_title="Reconhecimento Facial", layout="wide")

# =========================
# LOGIN SIMPLES (EM MEMÓRIA)
# =========================
def check_login():
    if "auth" not in st.session_state:
        st.session_state.auth = False

    st.sidebar.title("Login")

    user = st.sidebar.text_input("Usuário")
    pwd = st.sidebar.text_input("Senha", type="password")

    if st.sidebar.button("Entrar"):
        if user == "admin" and hashlib.sha256(pwd.encode()).hexdigest() == hashlib.sha256("1234".encode()).hexdigest():
            st.session_state.auth = True
        else:
            st.sidebar.error("Usuário ou senha inválidos")

    return st.session_state.auth


if not check_login():
    st.stop()

# =========================
# CONFIGURAÇÕES
# =========================
DB_FILE = "faces_db.npz"
THRESHOLD = 0.65

# =========================
# FUNÇÕES
# =========================
def load_db():
    if os.path.exists(DB_FILE):
        return dict(np.load(DB_FILE, allow_pickle=True))
    return {}

def save_db(db):
    np.savez(DB_FILE, **db)

def detectar_rostos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cascade.detectMultiScale(gray, 1.1, 5)

def histograma(face):
    hist = cv2.calcHist([face], [0, 1, 2], None,
                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

db = load_db()

# =========================
# DASHBOARD
# =========================
st.sidebar.header("📊 Dashboard")
st.sidebar.write(f"Rostos cadastrados: {len(db)}")

if st.sidebar.button("🗑️ Limpar banco"):
    db.clear()
    save_db(db)
    st.sidebar.success("Banco limpo")

# =========================
# CADASTRO
# =========================
st.header("🅱️ Cadastro de rosto")

name = st.text_input("Nome da pessoa")
file = st.file_uploader("Imagem para cadastro", type=["jpg", "jpeg", "png"])

if file and name:
    img = cv2.cvtColor(
        np.array(Image.open(file).convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

    faces = detectar_rostos(img)

    if len(faces) == 0:
        st.error("Nenhum rosto detectado")
    else:
        x, y, w, h = faces[0]
        face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
        db[name] = histograma(face)
        save_db(db)
        st.success("Rosto cadastrado com sucesso")
        st.image(face, channels="BGR", width=200)

# =========================
# WEBCAM - RECONHECIMENTO
# =========================
st.header("🎥 Reconhecimento em tempo real")

class Processor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = detectar_rostos(img)

        for (x, y, w, h) in faces:
            face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            hist = histograma(face)

            best_name = "Desconhecido"
            best_score = 0

            for name, ref in db.items():
                score = cv2.compareHist(hist, ref, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_name = name
                    best_score = score

            label = best_name if best_score > THRESHOLD else "Desconhecido"

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                img,
                f"{label} {best_score*100:.1f}%",
