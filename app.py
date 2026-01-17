LIMIAR = 0.7
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial", layout="centered")
st.title("🧠 Reconhecimento Facial (Upload de Imagem)")

DB_FILE = "face_db.npz"

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

db = load_db()

# =====================
# FUNÇÕES DE VISÃO
# =====================

def detectar_rostos(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    rostos = []
    for (x, y, w, h) in faces:
        rosto = img_bgr[y:y+h, x:x+w]
        rostos.append((rosto, (x, y, w, h)))

    return rostos

def extrair_histograma(rosto):
    hist = cv2.calcHist(
        [rosto], [0, 1, 2], None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )
    cv2.normalize(hist, hist)
    return hist

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
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    rostos = detectar_rostos(img_bgr)

    if not rostos:
        st.error("❌ Nenhum rosto detectado.")
    else:
        rosto, (x, y, w, h) = rostos[0]
        hist = extrair_histograma(rosto)
        db[nome] = hist
        save_db(db)

        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        st.success(f"✅ Rosto de '{nome}' cadastrado!")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=300)

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento facial")

arquivo2 = st.file_uploader(
    "Envie uma imagem para reconhecimento",
    type=["jpg", "jpeg", "png"],
    key="reconhecer"
)

if arquivo2:
    img = Image.open(arquivo2).convert("RGB")
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    rostos = detectar_rostos(img_bgr)

    if not rostos:
        st.error("❌ Nenhum rosto detectado.")
    elif not db:
        st.warning("⚠️ Nenhum rosto cadastrado ainda.")
    else:
        for rosto, (x, y, w, h) in rostos:
            hist = extrair_histograma(rosto)

            melhor_nome = "Desconhecido"
            melhor_score = 0

            for nome_db, hist_db in db.items():
                score = cv2.compareHist(hist, hist_db, cv2.HISTCMP_CORREL)
                if score > melhor_score:
                    melhor_score = score
                    if melhor_score >= LIMIAR:
                        melhor_nome = nome_db
                    else:
                        melhor_nome = "Desconhecido"


            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                f"{melhor_nome} ({melhor_score*100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=300)
        
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
        with col1:
            st.write(nome)
        with col2:
            if st.button("❌", key=f"del_{nome}"):
                del db[nome]
                save_db(db)
                st.rerun()
                
# =====================
# WEBCAM AO VIVO
# =====================

st.divider()
st.header("🎥 Webcam ao vivo (opcional)")

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            rostos = detectar_rostos(img)

            if db and rostos:
                for rosto, (x, y, w, h) in rostos:
                    hist = extrair_histograma(rosto)

                    melhor_nome = "Desconhecido"
                    melhor_score = 0

                    for nome_db, hist_db in db.items():
                        score = cv2.compareHist(
                            hist, hist_db, cv2.HISTCMP_CORREL
                        )
                        if score > melhor_score:
                            melhor_score = score
                            if melhor_score >= LIMIAR:
                                melhor_nome = nome_db
                            else:
                                melhor_nome = "Desconhecido"


                    cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2
                    )
                    cv2.putText(
                        img,
                        f"{melhor_nome} ({melhor_score*100:.1f}%)",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

except Exception as e:
    st.warning(
        "⚠️ Webcam indisponível neste ambiente. "
        "Use localmente com `streamlit run app.py`."
    )
