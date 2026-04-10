"""
Skin Lesion Classification API
Model  : DenseNet-201  (TTA F1-W = 0.8608, AUC = 0.9806)
Classes: 14 skin conditions
"""

import os
import io
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_model, predict, CLASS_NAMES, NUM_CLASSES

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT",
    "./best_model_phase2.pth"   # default: same directory as main.py
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_UPLOAD_MB = 10

# ── Global model state ────────────────────────────────────────────────────────
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info(f"Loading DenseNet-201 from {CHECKPOINT_PATH} on {DEVICE} …")
    t0 = time.time()
    try:
        app_state["model"]  = load_model(CHECKPOINT_PATH, DEVICE)
        app_state["device"] = DEVICE
        logger.info(f"Model loaded in {time.time() - t0:.1f}s  device={DEVICE}")
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        raise RuntimeError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
            "Set the MODEL_CHECKPOINT environment variable to the correct path."
        )
    yield
    app_state.clear()
    logger.info("Model released.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Skin Lesion Classifier",
    description = (
        "DenseNet-201 trained on 14 skin lesion classes.\n\n"
        "**Performance:** TTA F1-W = 0.8608 | AUC = 0.9806 | Accuracy = 0.8582\n\n"
        "Upload a skin lesion image to receive a predicted diagnosis with "
        "confidence scores for all 14 classes."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ──────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class    : str
    confidence         : float
    class_probabilities: dict[str, float]
    inference_time_ms  : float
    tta_used           : bool
    model              : str
    device             : str


class HealthResponse(BaseModel):
    status     : str
    model      : str
    device     : str
    num_classes: int
    classes    : list[str]


class ErrorResponse(BaseModel):
    error  : str
    detail : str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _validate_image(file: UploadFile, data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."
        )
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a JPEG or PNG image."
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                         # catches corrupt files
        img = Image.open(io.BytesIO(data))   # reopen after verify (verify closes stream)
        return img
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Could not read image: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Skin Lesion Classification API",
        "docs"   : "/docs",
        "health" : "/health",
        "predict": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns model status, device, and available classes.",
)
def health():
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status      = "ok",
        model       = "DenseNet-201",
        device      = app_state["device"],
        num_classes = NUM_CLASSES,
        classes     = CLASS_NAMES,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict skin lesion class",
    description=(
        "Upload a skin lesion image (JPEG / PNG / BMP, max 10 MB).\n\n"
        "Returns the predicted class, confidence percentage, and "
        "probability scores for all 14 classes.\n\n"
        "Set `tta=false` to skip Test-Time Augmentation for faster inference."
    ),
    responses={
        200: {"description": "Successful prediction"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        422: {"model": ErrorResponse, "description": "Corrupt or unreadable image"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_endpoint(
    file: UploadFile = File(..., description="Skin lesion image file"),
    tta : bool       = Query(True, description="Use Test-Time Augmentation (4 views averaged)"),
):
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    data  = await file.read()
    image = _validate_image(file, data)

    t0     = time.time()
    result = predict(
        model     = app_state["model"],
        pil_image = image,
        device    = app_state["device"],
        use_tta   = tta,
    )
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    logger.info(
        f"Predicted '{result['predicted_class']}' "
        f"({result['confidence']:.1f}%)  "
        f"tta={tta}  {elapsed_ms}ms"
    )

    return PredictionResponse(
        predicted_class     = result["predicted_class"],
        confidence          = result["confidence"],
        class_probabilities = result["class_probabilities"],
        inference_time_ms   = elapsed_ms,
        tta_used            = tta,
        model               = "DenseNet-201",
        device              = app_state["device"],
    )


@app.post(
    "/predict/top3",
    summary="Predict — top 3 classes only",
    description="Same as /predict but returns only the top 3 most likely classes.",
)
async def predict_top3(
    file: UploadFile = File(...),
    tta : bool       = Query(True),
):
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    data   = await file.read()
    image  = _validate_image(file, data)

    t0     = time.time()
    result = predict(app_state["model"], image, app_state["device"], tta)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    # Sort all probabilities and return top 3
    top3 = sorted(
        result["class_probabilities"].items(),
        key=lambda x: x[1], reverse=True
    )[:3]

    return JSONResponse({
        "predicted_class" : result["predicted_class"],
        "confidence"      : result["confidence"],
        "top3"            : [{"class": c, "confidence": round(p, 2)} for c, p in top3],
        "inference_time_ms": elapsed_ms,
        "tta_used"        : tta,
    })
