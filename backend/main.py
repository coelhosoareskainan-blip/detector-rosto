from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from io import BytesIO

from face_engine import extract_embeddings, recognize_face
from database import load_db, save_db

app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = load_db()


@app.post("/register")
async def register(name: str, file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img_np = np.array(img)

    embeddings = extract_embeddings(img_np)

    db.setdefault(name, []).extend(embeddings)
    save_db(db)

    return {"status": "ok", "faces": len(embeddings)}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img_np = np.array(img)

    results = recognize_face(img_np, db)
    return {"results": results}
