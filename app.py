import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import os

st.set_page_config(page_title="Reconhecimento Facial", layout="centered")
st.title("Reconhecimento Facial • Webcam + Banco")

DB_FILE = "faces_db.npz"

def load_db():
    if os.path.exists(DB_FILE):
        data = np.load(DB_FILE, allow_pickle=True)
        return dict(data)
    return {}

def save_db(db):
    np.savez(DB_FILE, **db)

def detectar_rosto(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    rosto = img[y:y+h, x:x+w]
    return cv2.resize(rosto, (200, 200))

def histograma(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

db = load_db()

st.header("🅱️ Cadastro de rosto")
nome = st.text_input("Nome da pessoa")
img_file = st.file_uploader("Imagem para cadastro", type=["jpg", "jpeg", "png"])

if img_file and nome:
    img = np.array(Image.open(img_file).convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rosto = detectar_rosto(img)

    if rosto is None:
        st.error("Nenhum rosto detectado.")
    else:
        db[nome] = histograma(rosto)
        save_db(db)
        st.success(f"Rosto de '{nome}' cadastrado com sucesso!")
        st.image(rosto, channels="BGR", width=200)

st.divider()
st.header("🅰️ Webcam • Reconhecimento em tempo real")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rosto = detectar_rosto(img)

        if rosto is not None and db:
            hist = histograma(rosto)
            best_name = "Desconhecido"
            best_score = 0

            for name, ref_hist in db.items():
                score = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_score = score
                    best_name = name

            label = f"{best_name} ({best_score*100:.1f}%)"
            cv2.putText(img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="face-recognition",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
