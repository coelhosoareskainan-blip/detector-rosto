from services.face_api import get_embeddings
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

# =====================
# CONFIG
# =====================

st.set_page_config(page_title="Reconhecimento Facial IA", layout="centered")
st.title("🧠 Reconhecimento Facial")

DB_FILE = "data/faces.json"

LIMIAR = 0.55          # 🔥 ajustado para Face++
CONF_MINIMA = 45.0     # 🔥 ajustado para Face++

# =====================
# UTILIDADES
# =====================

def load_db():
    try:
        import json
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_db(db):
    import json
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

db = load_db()

# =====================
# DETECTOR
# =====================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detectar_rostos(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(90, 90)
    )

def crop_face(img, box):
    x, y, w, h = box
    face = img[y:y+h, x:x+w]
    buf = BytesIO()
    Image.fromarray(face).save(buf, format="JPEG")
    buf.seek(0)
    return buf

def distancia(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# =====================
# CADASTRO
# =====================

st.header("🅱️ Cadastro")

nome = st.text_input("Nome")
arquivo = st.file_uploader("Imagem para cadastro", ["jpg","png","jpeg"])

if arquivo and nome:
    img = np.array(Image.open(arquivo).convert("RGB"))
    faces = detectar_rostos(img)

    if len(faces) == 0:
        st.error("Nenhum rosto detectado")
    else:
        face_file = crop_face(img, faces[0])
        vecs = get_embeddings(face_file)

        if not vecs:
            st.error("IA não retornou embedding")
        else:
            db[nome] = vecs[0]
            save_db(db)
            st.success(f"{nome} cadastrado com sucesso")

# =====================
# RECONHECIMENTO
# =====================

st.divider()
st.header("🅰️ Reconhecimento")

arquivo2 = st.file_uploader("Imagem para reconhecimento", ["jpg","png","jpeg"], key="rec")

if arquivo2 and db:
    img = np.array(Image.open(arquivo2).convert("RGB"))
    faces = detectar_rostos(img)
    
for (x, y, w, h) in faces:

    face_file = crop_face(img_np, (x, y, w, h))
    embeddings = get_embeddings(face_file)

    if not embeddings:
        label = "DESCONHECIDO"
        cor = (255, 0, 0)
    else:
        emb = embeddings[0]

        melhor_nome = "Desconhecido"
        melhor_dist = float("inf")

        for nome_db, emb_db in db.items():
            d = distancia(emb, emb_db)
            if d < melhor_dist:
                melhor_dist = d
                melhor_nome = nome_db

        confianca = max(0, (1 - melhor_dist / LIMIAR)) * 100

        if melhor_dist <= LIMIAR and confianca >= CONF_MINIMA:
            label = f"{melhor_nome} ({confianca:.1f}%)"
            cor = (0, 255, 0)
            nome_final = melhor_nome
        else:
            label = "DESCONHECIDO"
            cor = (255, 0, 0)
            nome_final = "Desconhecido"
            confianca = 0.0

    # 🔒 posição segura do texto
    texto_y = y - 10 if y - 10 > 20 else y + h + 25

    cv2.rectangle(img_np, (x, y), (x+w, y+h), cor, 2)
    cv2.putText(
        img_np,
        label,
        (x, texto_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        cor,
        2
    )

    history.append({
        "nome": nome_final,
        "confianca": round(confianca, 1),
        "distancia": round(melhor_dist if melhor_dist != float("inf") else 0, 3),
        "data": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    })
