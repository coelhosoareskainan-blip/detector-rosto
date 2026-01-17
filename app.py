import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Detector de Rosto")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img_array.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            bw, bh = int(bbox.width * w), int(bbox.height * h)

            cv2.rectangle(img_array, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        st.success("Rosto detectado com sucesso!")
    else:
        st.warning("Nenhum rosto detectado.")

    st.image(img_array, channels="BGR")

