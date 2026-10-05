"""FastAPI inference service for the diabetic retinopathy screening pipeline."""

import base64
from functools import lru_cache
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

import stage7_pipeline as s7

app = FastAPI(title="Explainable DR Screening API")


@lru_cache(maxsize=1)
def load_models():
    return s7.load_models(
        "models/best_model.pth",
        "models/seg_model.pth",
        "models/seg_model_v2.pth",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(file.file).convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid image file.") from exc

    cls_model, seg_models = load_models()
    result = s7.run_pipeline(image, cls_model, seg_models)
    response = {
        "status": result["status"],
        "quality": result["quality"],
        "report": result["report"],
    }

    if result["status"] == "graded":
        overlay = Image.fromarray(result["overlay"])
        buffer = BytesIO()
        overlay.save(buffer, format="PNG")
        response.update({
            "severity": result["severity"],
            "confidence": result["confidence"],
            "probabilities": result["probs"],
            "recommendation": result["recommendation"],
            "evidence": result["evidence"],
            "overlay_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        })

    return response
