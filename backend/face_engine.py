import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
DIST_THRESHOLD = 0.35


def extract_embeddings(img_np):
    reps = DeepFace.represent(
        img_path=img_np,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,
        align=True
    )
    return [r["embedding"] for r in reps]


def recognize_face(img_np, database):
    detections = DeepFace.extract_faces(
        img_path=img_np,
        detector_backend=DETECTOR,
        enforce_detection=False,
        align=True
    )

    results = []

    for det in detections:
        face_img = det["face"]
        region = det["facial_area"]

        rep = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend="skip"
        )[0]["embedding"]

        best_name = "DESCONHECIDO"
        best_dist = 1.0

        for name, embs in database.items():
            for emb_db in embs:
                cos = np.dot(rep, emb_db) / (
                    np.linalg.norm(rep) * np.linalg.norm(emb_db)
                )
                dist = 1 - cos

                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        results.append({
            "name": best_name if best_dist <= DIST_THRESHOLD else "DESCONHECIDO",
            "distance": best_dist,
            "region": region
        })

    return results
