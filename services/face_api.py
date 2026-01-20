import requests
import numpy as np
import os

# =====================
# CONFIG FACE++
# =====================

API_KEY = os.getenv("FACEPP_API_KEY")
API_SECRET = os.getenv("FACEPP_API_SECRET")

API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

# =====================================================
# FUNÇÃO NOVA — MULTI-ROSTO (SUBSTITUI A ANTIGA)
# =====================================================

def get_embeddings(uploaded_file):
    """
    Retorna uma LISTA de vetores normalizados
    Um vetor por rosto detectado
    """

    # 1. validar credenciais
    if not API_KEY or not API_SECRET:
        return []

    # 2. montar request
    files = {
        "image_file": uploaded_file.getvalue()
    }

    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_landmark": 1
    }

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=15)
    except Exception:
        return []

    if response.status_code != 200:
        return []

    result = response.json()

    # 3. validar resposta
    if "faces" not in result or len(result["faces"]) == 0:
        return []

    embeddings = []

    # 4. extrair landmarks de TODOS os rostos
    for face in result["faces"]:
        if "landmark" not in face:
            continue

        landmarks = face["landmark"]

        embedding = []
        for point in landmarks.values():
            embedding.append(float(point["x"]))
            embedding.append(float(point["y"]))

        if len(embedding) == 0:
            continue

        vec = np.array(embedding, dtype=np.float32)

        # =====================
        # NORMALIZAÇÃO (ESSENCIAL)
        # =====================
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue

        vec = vec / norm
        embeddings.append(vec.tolist())

    return embeddings
