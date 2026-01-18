import requests
import numpy as np
import os

API_KEY = os.getenv("FACEPP_API_KEY")
API_SECRET = os.getenv("FACEPP_API_SECRET")

API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"

def get_embedding(uploaded_file):
    if API_KEY is None or API_SECRET is None:
        return None

    files = {
        "image_file": uploaded_file.getvalue()
    }

    data = {
        "api_key": API_KEY,
        "api_secret": API_SECRET,
        "return_attributes": "none"
    }

    response = requests.post(API_URL, files=files, data=data)

    if response.status_code != 200:
        return None

    result = response.json()

    if "faces" not in result or len(result["faces"]) == 0:
        return None

    # Face++ não retorna embedding direto,
    # então usamos landmarks como vetor simplificado
    landmarks = result["faces"][0]["landmark"]

    embedding = []
    for point in landmarks.values():
        embedding.append(point["x"])
        embedding.append(point["y"])

    vec = np.array(embedding, dtype=np.float32)

# normalização (ESSENCIAL)
norm = np.linalg.norm(vec)
if norm == 0:
    return None

vec = vec / norm
return vec.tolist()
