"""
FastAPI server for ISL sign language recognition.

Endpoints:
  - POST /predict           – CNN inference on a base64 image (letter)
  - POST /predict_landmarks – MLP inference on 126 landmark floats (letter)
  - POST /predict_word      – LSTM inference on sequence of frames (word)
  - POST /predict_sentence  – LSTM inference on sequence of frames (sentence)
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared application state (populated by lifespan)
# ---------------------------------------------------------------------------
app_state: dict[str, Any] = {}

MODEL_PATH      = "sign_image_model.h5"
CLASSES_PATH    = "classes.txt"
IMG_SIZE        = 64

LANDMARK_MODEL_PATH   = "sign_landmark_model.keras"
LANDMARK_CLASSES_PATH = "landmark_classes.txt"

WORD_MODEL_PATH       = "sign_word_model.keras"
WORD_CLASSES_PATH     = "word_classes.txt"

SENTENCE_MODEL_PATH   = "sign_sentence_model.keras"
SENTENCE_CLASSES_PATH = "sentence_classes.txt"

SIGN_GLOSSES_PATH     = r"D:\archive\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL Corpus sign glosses.csv"

PHRASE_MODEL_PATH     = "sign_phrase_model.keras"
PHRASE_CLASSES_PATH   = "phrase_classes.txt"


# ---------------------------------------------------------------------------
# Lifespan – load model & classes once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts before accepting requests; clean up on shutdown."""
    # Validate required files exist
    if not os.path.isfile(MODEL_PATH):
        logger.error(
            "Model file not found: %s — run train.py first to generate it.", MODEL_PATH
        )
        raise SystemExit(1)

    if not os.path.isfile(CLASSES_PATH):
        logger.error(
            "Classes file not found: %s — run train.py first to generate it.", CLASSES_PATH
        )
        raise SystemExit(1)

    # Lazy TF import (heavy; keep outside module-level to speed up startup)
    import tensorflow as tf

    logger.info("Loading CNN model from %s …", MODEL_PATH)
    app_state["model"] = tf.keras.models.load_model(MODEL_PATH)
    logger.info("CNN model loaded successfully.")

    with open(CLASSES_PATH, "r", encoding="utf-8") as fh:
        app_state["classes"] = [line.strip() for line in fh if line.strip()]
    logger.info("Loaded %d class labels.", len(app_state["classes"]))

    # Landmark model – optional
    if os.path.isfile(LANDMARK_MODEL_PATH) and os.path.isfile(LANDMARK_CLASSES_PATH):
        logger.info("Loading landmark MLP model from %s …", LANDMARK_MODEL_PATH)
        app_state["landmark_model"] = tf.keras.models.load_model(LANDMARK_MODEL_PATH)
        with open(LANDMARK_CLASSES_PATH, "r", encoding="utf-8") as fh:
            app_state["landmark_classes"] = [l.strip() for l in fh if l.strip()]
        logger.info("Landmark model loaded. Classes: %d", len(app_state["landmark_classes"]))
    else:
        logger.info("Landmark model not found – /predict_landmarks will return 503 until trained.")

    # Word model – optional
    if os.path.isfile(WORD_MODEL_PATH) and os.path.isfile(WORD_CLASSES_PATH):
        logger.info("Loading word LSTM model from %s …", WORD_MODEL_PATH)
        app_state["word_model"] = tf.keras.models.load_model(WORD_MODEL_PATH)
        with open(WORD_CLASSES_PATH, "r", encoding="utf-8") as fh:
            app_state["word_classes"] = [l.strip() for l in fh if l.strip()]
        logger.info("Word model loaded. Classes: %d", len(app_state["word_classes"]))
    else:
        logger.info("Word model not found – run train_words.py first.")

    # Sentence model – optional
    if os.path.isfile(SENTENCE_MODEL_PATH) and os.path.isfile(SENTENCE_CLASSES_PATH):
        logger.info("Loading sentence LSTM model from %s …", SENTENCE_MODEL_PATH)
        app_state["sentence_model"] = tf.keras.models.load_model(SENTENCE_MODEL_PATH)
        with open(SENTENCE_CLASSES_PATH, "r", encoding="utf-8") as fh:
            app_state["sentence_classes"] = [l.strip() for l in fh if l.strip()]
        logger.info("Sentence model loaded. Classes: %d", len(app_state["sentence_classes"]))
    else:
        logger.info("Sentence model not found – run train_sentences.py first.")

    # Sign glosses – optional but useful
    if os.path.isfile(SIGN_GLOSSES_PATH):
        import csv
        glosses: dict[str, str] = {}
        with open(SIGN_GLOSSES_PATH, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sentence = row.get("Sentence", "").strip().lower()
                gloss    = row.get("SIGN GLOSSES", "").strip()
                if sentence and gloss:
                    glosses[sentence] = gloss
        app_state["sign_glosses"] = glosses
        logger.info("Sign glosses loaded: %d entries", len(glosses))
    else:
        app_state["sign_glosses"] = {}
        logger.info("Sign glosses file not found – gloss field will be empty.")

    # Phrase model – optional
    if os.path.isfile(PHRASE_MODEL_PATH) and os.path.isfile(PHRASE_CLASSES_PATH):
        logger.info("Loading phrase MLP model from %s …", PHRASE_MODEL_PATH)
        app_state["phrase_model"] = tf.keras.models.load_model(PHRASE_MODEL_PATH)
        with open(PHRASE_CLASSES_PATH, "r", encoding="utf-8") as fh:
            app_state["phrase_classes"] = [l.strip() for l in fh if l.strip()]
        logger.info("Phrase model loaded. Classes: %d", len(app_state["phrase_classes"]))
    else:
        logger.info("Phrase model not found – run train_phrases.py first.")

    yield  # server runs here

    # Shutdown – release references
    app_state.clear()
    logger.info("Model resources released.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Sign Language Recognition API", lifespan=lifespan)

# CORS – required for mobile / cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (CSS, JS, images, etc.)
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    if os.path.isdir(os.path.join("static", "assets")):
        app.mount("/assets", StaticFiles(directory=os.path.join("static", "assets")), name="assets")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    image: str  # base64-encoded image (JPEG / PNG)


class PredictResponse(BaseModel):
    character: str
    confidence: float


class LandmarkRequest(BaseModel):
    landmarks: list  # list of 126 floats (2 hands × 21 landmarks × 3)


class LandmarkResponse(BaseModel):
    character: str
    confidence: float


class SequenceRequest(BaseModel):
    """Sequence of landmark frames for word/sentence recognition.
    frames: list of N frames, each frame is a list of 126 floats.
    """
    frames: list


class SequenceResponse(BaseModel):
    label: str
    confidence: float
    gloss: str = ""   # ISL sign gloss sequence (sentence model only)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    html_path = os.path.join("static", "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Accept a base64-encoded image, run CNN inference, return the top-1 prediction.

    - 400 if the base64 payload cannot be decoded or is not a valid image.
    - 500 if inference fails for an unexpected reason.
    """
    # --- Decode base64 ---
    try:
        image_bytes = base64.b64decode(request.image, validate=True)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    # --- Pre-process ---
    try:
        pil_image = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img_array = np.array(pil_image, dtype=np.float32) / 255.0  # normalise [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pre-processing error: {exc}") from exc

    # --- Inference ---
    try:
        model = app_state["model"]
        classes = app_state["classes"]
        predictions = model.predict(img_array, verbose=0)  # shape (1, num_classes)
        class_idx = int(np.argmax(predictions[0]))
        confidence = float(round(float(predictions[0][class_idx]), 4))
        character = classes[class_idx]
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return PredictResponse(character=character, confidence=confidence)


@app.post("/predict_landmarks", response_model=LandmarkResponse)
async def predict_landmarks(request: LandmarkRequest):
    """
    Accept 63 normalised landmark floats [x0,y0,z0, …, x20,y20,z20],
    run MLP inference, return the top-1 prediction.

    - 400 if the landmark list doesn't have exactly 63 floats.
    - 503 if the landmark model hasn't been loaded yet.
    - 500 if inference fails.
    """
    if "landmark_model" not in app_state:
        raise HTTPException(
            status_code=503,
            detail="Landmark model not loaded. Run train_landmarks.py first.",
        )

    # Validate input length
    if len(request.landmarks) not in (63, 126):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 126 landmark values (both hands), got {len(request.landmarks)}.",
        )

    try:
        n = len(request.landmarks)
        features = np.array(request.landmarks, dtype=np.float32).reshape(1, n)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid landmark data: {exc}") from exc

    try:
        model   = app_state["landmark_model"]
        classes = app_state["landmark_classes"]
        preds   = model.predict(features, verbose=0)  # shape (1, num_classes)
        idx     = int(np.argmax(preds[0]))
        confidence = float(round(float(preds[0][idx]), 4))
        character  = classes[idx]
    except Exception as exc:
        logger.exception("Landmark inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return LandmarkResponse(character=character, confidence=confidence)


@app.post("/predict_word", response_model=SequenceResponse)
async def predict_word(request: SequenceRequest):
    """
    Accept a sequence of landmark frames and return the predicted ISL word.
    Each frame is a list of 126 floats (2 hands × 21 landmarks × 3).
    """
    if "word_model" not in app_state:
        raise HTTPException(status_code=503,
                            detail="Word model not loaded. Run train_words.py first.")

    SEQ_LEN      = 10
    FEATURE_SIZE = 126

    try:
        frames = [np.array(f, dtype=np.float32) for f in request.frames]
        seq    = np.array(frames, dtype=np.float32)

        # Pad or truncate to SEQ_LEN
        if len(seq) >= SEQ_LEN:
            indices = np.linspace(0, len(seq)-1, SEQ_LEN, dtype=int)
            seq = seq[indices]
        else:
            pad = np.zeros((SEQ_LEN - len(seq), FEATURE_SIZE), dtype=np.float32)
            seq = np.vstack([seq, pad])

        seq = seq.reshape(1, SEQ_LEN, FEATURE_SIZE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sequence data: {exc}")

    try:
        preds      = app_state["word_model"].predict(seq, verbose=0)
        idx        = int(np.argmax(preds[0]))
        confidence = float(round(float(preds[0][idx]), 4))
        label      = app_state["word_classes"][idx]
    except Exception as exc:
        logger.exception("Word inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return SequenceResponse(label=label, confidence=confidence)


@app.get("/glosses")
async def get_glosses():
    """
    Return all sentence → ISL sign gloss mappings.
    Useful for the frontend to display the gloss alongside predictions.
    """
    return app_state.get("sign_glosses", {})


@app.post("/predict_sentence", response_model=SequenceResponse)
async def predict_sentence(request: SequenceRequest):
    """
    Accept a sequence of landmark frames and return the predicted ISL sentence.
    Each frame is a list of 126 floats (2 hands × 21 landmarks × 3).
    """
    if "sentence_model" not in app_state:
        raise HTTPException(status_code=503,
                            detail="Sentence model not loaded. Run train_sentences.py first.")

    SEQ_LEN      = 30
    FEATURE_SIZE = 126

    try:
        frames = [np.array(f, dtype=np.float32) for f in request.frames]
        seq    = np.array(frames, dtype=np.float32)

        if len(seq) >= SEQ_LEN:
            indices = np.linspace(0, len(seq)-1, SEQ_LEN, dtype=int)
            seq = seq[indices]
        else:
            pad = np.zeros((SEQ_LEN - len(seq), FEATURE_SIZE), dtype=np.float32)
            seq = np.vstack([seq, pad])

        seq = seq.reshape(1, SEQ_LEN, FEATURE_SIZE)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sequence data: {exc}")

    try:
        preds      = app_state["sentence_model"].predict(seq, verbose=0)
        idx        = int(np.argmax(preds[0]))
        confidence = float(round(float(preds[0][idx]), 4))
        label      = app_state["sentence_classes"][idx]
        gloss      = app_state.get("sign_glosses", {}).get(label.lower(), "")
    except Exception as exc:
        logger.exception("Sentence inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return SequenceResponse(label=label, confidence=confidence, gloss=gloss)


@app.post("/predict_phrase", response_model=LandmarkResponse)
async def predict_phrase(request: LandmarkRequest):
    """
    Accept 126 normalised landmark floats (2 hands × 21 landmarks × 3),
    run phrase MLP inference, return the top-1 prediction.

    Same input format as /predict_landmarks but uses the phrase-specific model
    trained on the 44-class phrase/word dataset.

    - 503 if the phrase model hasn't been loaded yet.
    - 400 if landmark count is invalid.
    - 500 if inference fails.
    """
    if "phrase_model" not in app_state:
        raise HTTPException(
            status_code=503,
            detail="Phrase model not loaded. Run train_phrases.py first.",
        )

    if len(request.landmarks) not in (63, 126):
        raise HTTPException(
            status_code=400,
            detail=f"Expected 126 landmark values (both hands), got {len(request.landmarks)}.",
        )

    try:
        n        = len(request.landmarks)
        features = np.array(request.landmarks, dtype=np.float32).reshape(1, n)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid landmark data: {exc}") from exc

    try:
        model      = app_state["phrase_model"]
        classes    = app_state["phrase_classes"]
        preds      = model.predict(features, verbose=0)
        idx        = int(np.argmax(preds[0]))
        confidence = float(round(float(preds[0][idx]), 4))
        character  = classes[idx]
    except Exception as exc:
        logger.exception("Phrase inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return LandmarkResponse(character=character, confidence=confidence)
