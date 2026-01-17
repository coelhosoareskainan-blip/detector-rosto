import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Detector de Rosto")

uploaded_file = st.file_uploader(
    "Envie uma imagem",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Corrige imagens PNG com canal alpha
    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # ✅ NOME CORRETO: face_mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    results = face_mesh.process(img_rgb)

    h, w, _ = img_array.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            cv2.rectangle(
                img_array,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

        st.success("Rosto detectado com sucesso!")
    else:
        st.warning("Nenhum rosto detectado.")

    st.image(img_array, channels="BGR")
