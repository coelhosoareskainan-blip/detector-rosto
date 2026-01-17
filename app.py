import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Detector de Rosto (Estável)")

uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(
                img_array,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        st.success("Rosto detectado com sucesso!")
    else:
        st.warning("Nenhum rosto detectado.")

    st.image(img_array, channels="RGB")
