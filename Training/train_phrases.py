"""
Train an MLP on MediaPipe hand landmarks extracted from phrase/word images.

Dataset: D:\\Downloads\\images for phrases\\images for phrases\\<LABEL>\\*.png
         (44 classes, 40 images each = 1,760 images)

Outputs:
  sign_phrase_model.keras   – trained Keras model
  phrase_classes.txt        – ordered class names (one per line)

Usage:
  python train_phrases.py
"""

from __future__ import annotations

import os
import sys
import logging
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = r"D:\Downloads\images for phrases\images for phrases"
MODEL_OUT    = os.path.join(SCRIPT_DIR, "sign_phrase_model.keras")
CLASSES_OUT  = os.path.join(SCRIPT_DIR, "phrase_classes.txt")
TASK_FILE    = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FEATURE_SIZE = 126   # 2 hands × 21 landmarks × 3 (x,y,z)


# ── Ensure task file ──────────────────────────────────────────────────────
def ensure_task_file():
    import urllib.request
    if os.path.isfile(TASK_FILE) and os.path.getsize(TASK_FILE) > 100_000:
        logger.info("Task file found: %s", TASK_FILE)
        return
    logger.info("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)
    logger.info("Downloaded → %.1f MB", os.path.getsize(TASK_FILE) / 1e6)


# ── MediaPipe detector ────────────────────────────────────────────────────
def make_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=TASK_FILE),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(opts)


# ── Normalize one hand (mirrors JS frontend + train_landmarks.py) ─────────
def normalize_hand(lm_list) -> np.ndarray:
    xs = np.array([p.x for p in lm_list], dtype=np.float32)
    ys = np.array([p.y for p in lm_list], dtype=np.float32)
    zs = np.array([p.z for p in lm_list], dtype=np.float32)
    min_x, min_y = xs.min(), ys.min()
    scale = max(xs.max() - min_x, ys.max() - min_y) or 1.0
    xs = (xs - min_x) / scale
    ys = (ys - min_y) / scale
    out = np.empty(63, dtype=np.float32)
    for i in range(21):
        out[i * 3], out[i * 3 + 1], out[i * 3 + 2] = xs[i], ys[i], zs[i]
    return out


# ── Extract landmarks from one image ─────────────────────────────────────
def extract_landmarks(img_path: str, detector) -> np.ndarray | None:
    """Returns (126,) feature vector or None if no hand detected."""
    import mediapipe as mp
    img = cv2.imread(img_path)
    if img is None:
        return None
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for lm, hand in zip(result.hand_landmarks, result.handedness):
        label = hand[0].category_name   # "Left" or "Right"
        slot  = 0 if label == "Left" else 63
        features[slot: slot + 63] = normalize_hand(lm)
    return features


# ── Build dataset ─────────────────────────────────────────────────────────
def build_dataset():
    class_names = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not class_names:
        logger.error("No class folders found in %s", DATASET_DIR)
        sys.exit(1)

    logger.info("Found %d classes", len(class_names))

    X, y = [], []
    skipped = 0
    detector = make_detector()

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATASET_DIR, class_name)
        images = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        found = 0
        for fname in images:
            features = extract_landmarks(os.path.join(class_dir, fname), detector)
            if features is None:
                skipped += 1
                continue
            X.append(features)
            y.append(label_idx)
            found += 1

        logger.info("  %-20s  %d / %d usable", class_name, found, len(images))

    detector.close()
    logger.info("Total usable: %d  |  Skipped (no hand): %d", len(X), skipped)

    if not X:
        logger.error("No landmarks extracted. Check images.")
        sys.exit(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), class_names


# ── Model ─────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_SIZE,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="isl_phrase_mlp")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    logger.info("=== ISL Phrase/Word MLP Training ===")
    logger.info("Dataset: %s", DATASET_DIR)

    ensure_task_file()

    X, y, class_names = build_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Val: %d  |  Classes: %d", len(X_train), len(X_val), len(class_names))

    import tensorflow as tf
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Final validation accuracy: %.2f%%", val_acc * 100)

    model.save(MODEL_OUT)
    logger.info("Model saved → %s", MODEL_OUT)

    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names) + "\n")
    logger.info("Classes saved → %s  (%d phrases)", CLASSES_OUT, len(class_names))
    logger.info("Done!")


if __name__ == "__main__":
    main()
