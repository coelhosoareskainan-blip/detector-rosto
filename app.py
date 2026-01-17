import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Reconhecimento Facial Básico (Comparação)")

st.write("Envie duas imagens para comparar se os rostos são semelhantes.")

img1_file = st.file_uploader("Imagem 1", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Imagem 2", type=["jpg", "jpeg", "png"])

def detectar_rosto(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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

if img1_file and img2_file:
    img1 = np.array(Image.open(img1_file).convert("RGB"))
    img2 = np.array(Image.open(img2_file).convert("RGB"))

    rosto1 = detectar_rosto(img1)
    rosto2 = detectar_rosto(img2)

    if rosto1 is None or rosto2 is None:
        st.error("Não foi possível detectar rosto em uma das imagens.")
    else:
        hist1 = histograma(rosto1)
        hist2 = histograma(rosto2)

        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        similaridade = max(0, score) * 100

        st.image([rosto1, rosto2], caption=["Rosto 1", "Rosto 2"], width=200)

        st.write(f"### Similaridade: {similaridade:.2f}%")

        if similaridade > 70:
            st.success("Os rostos são semelhantes")
        else:
            st.warning("Os rostos são diferentes")
