import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os
from auth import login

st.set_page_config("Reconhecimento Facial", layout="wide")

if not login():
    st.stop()

st.title("Sistema de Reconhecimento Facial • MVP")

DB_FILE = "faces_db.npz"
THRESHOLD = 0.65

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
                        [8, 8, 8], [0, 256]*3)
    cv2.normalize(hist, hist)
    return hist

db = load_db()

# 🧑‍💻 DASHBOARD
st.sidebar.header("📊 Dashboard")
st.sidebar.write(f"Total de rostos: {len(db)}")

if st.sidebar.button("🗑️ Limpar banco"):
    db.clear()
    save_db(db)
    st.sidebar.success("Banco limpo")

# 🅱️ CADASTRO
st.header("🅱️ Cadastro de rosto")
name = st.text_input("Nome")
file = st.file_uploader("Imagem", type=["jpg", "jpeg", "png"])

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
        st.success("Rosto cadastrado!")
        st.image(face, channels="BGR", width=200)

# 🎥 WEBCAM
st.header("🎥 Reconhecimento em tempo real")

class Processor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        faces = detectar_rostos(img)

        for (x, y, w, h) in faces:
            face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
            hist = histograma(face)

            best, score = "Desconhecido", 0
            for name, ref in db.items():
                s = cv2.compareHist(hist, ref, cv2.HISTCMP_CORREL)
                if s > score:
                    best, score = name, s

            label = best if score > THRESHOLD else "Desconhecido"
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} {score*100:.1f}%",
                        (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="cam",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False}
)
