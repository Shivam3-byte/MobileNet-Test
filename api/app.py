from __future__ import annotations

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.classification.infer import predict_image

app = FastAPI(title="Plant Disease Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "models/checkpoints/mobilenetv3/field_finetune/best.pt")
UPLOAD_DIR = Path("api/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    destination = UPLOAD_DIR / file.filename
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not Path(CHECKPOINT_PATH).exists():
        raise HTTPException(status_code=503, detail="Checkpoint not found. Train or fine-tune the model first.")

    return predict_image(CHECKPOINT_PATH, str(destination))
