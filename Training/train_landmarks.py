"""
Train a lightweight MLP on MediaPipe hand landmarks extracted from Image_Data.
Supports BOTH hands (126 features = 2 × 21 landmarks × 3 coords).
If only one hand is detected, the second hand's features are zero-padded.

Uses the mediapipe.tasks API (mediapipe >= 0.10).
Requires hand_landmarker.task in the same directory (auto-downloaded if missing).

Usage:
    python train_landmarks.py

Outputs:
    sign_landmark_model.keras   – trained Keras model
    landmark_classes.txt        – ordered class names (one per line)
"""

from __future__ import annotations

import os
import sys
import logging
import urllib.request
import warnings

# Suppress TensorFlow / mediapipe noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
IMAGE_DATA_DIR = os.path.join(SCRIPT_DIR, "Image_Data")
MODEL_OUT      = os.path.join(SCRIPT_DIR, "sign_landmark_model.keras")
CLASSES_OUT    = os.path.join(SCRIPT_DIR, "landmark_classes.txt")
TASK_FILE      = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ---------------------------------------------------------------------------
# Ensure hand_landmarker.task is present
# ---------------------------------------------------------------------------
def ensure_task_file():
    if os.path.isfile(TASK_FILE) and os.path.getsize(TASK_FILE) > 100_000:
        logger.info("Task file found: %s", TASK_FILE)
        return
    logger.info("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)
    logger.info("Downloaded → %s (%.1f MB)", TASK_FILE, os.path.getsize(TASK_FILE) / 1e6)


# ---------------------------------------------------------------------------
# Landmark extraction using mediapipe.tasks
# ---------------------------------------------------------------------------
def make_detector():
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=TASK_FILE),
        num_hands=2,                        # ← detect up to 2 hands
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


FEATURE_SIZE = 126  # 2 hands × 21 landmarks × 3 coords (x, y, z)


def normalize_hand(lm_list) -> np.ndarray:
    """
    Normalize 21 landmarks to scale-invariant [0,1] range.
    Returns a (63,) float32 array.
    Must mirror normalizeLandmarks() in the JS frontend.
    """
    xs = np.array([p.x for p in lm_list], dtype=np.float32)
    ys = np.array([p.y for p in lm_list], dtype=np.float32)
    zs = np.array([p.z for p in lm_list], dtype=np.float32)

    min_x, min_y = xs.min(), ys.min()
    scale = max(xs.max() - min_x, ys.max() - min_y)
    if scale == 0:
        scale = 1.0

    xs = (xs - min_x) / scale
    ys = (ys - min_y) / scale

    features = np.empty(63, dtype=np.float32)
    for i in range(21):
        features[i * 3]     = xs[i]
        features[i * 3 + 1] = ys[i]
        features[i * 3 + 2] = zs[i]
    return features


def extract_landmarks(image_path: str, detector) -> np.ndarray | None:
    """
    Return a (126,) float32 array for BOTH hands.

    Layout: [left_hand_63_floats | right_hand_63_floats]
    - If a hand is not detected its 63 floats are zero-padded.
    - Handedness label from MediaPipe ("Left"/"Right") determines slot.
    - Returns None only if NO hand at all is found.

    Normalisation per hand (must mirror the JS frontend):
        1. Subtract min(x) from all x coords, subtract min(y) from all y coords.
        2. Divide by max(range_x, range_y) → scale-invariant [0, 1] range.
        3. z kept as-is (MediaPipe normalises z relative to wrist).
    """
    import mediapipe as mp

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    features = np.zeros(FEATURE_SIZE, dtype=np.float32)

    for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        slot  = 0 if label == "Left" else 63
        features[slot: slot + 63] = normalize_hand(hand_lm)

    return features


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------
def build_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    class_names: list[str] = sorted(
        d for d in os.listdir(IMAGE_DATA_DIR)
        if os.path.isdir(os.path.join(IMAGE_DATA_DIR, d))
    )

    if not class_names:
        logger.error("No class subdirectories found in %s", IMAGE_DATA_DIR)
        sys.exit(1)

    logger.info("Found %d classes: %s", len(class_names), class_names)

    X: list[np.ndarray] = []
    y: list[int]        = []
    skipped             = 0

    detector = make_detector()

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(IMAGE_DATA_DIR, class_name)
        images    = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        found = 0
        for fname in images:
            path     = os.path.join(class_dir, fname)
            features = extract_landmarks(path, detector)
            if features is None:
                skipped += 1
                continue
            X.append(features)
            y.append(label_idx)
            found += 1

        logger.info("  %-12s  %d / %d images usable", class_name, found, len(images))

    detector.close()
    logger.info("Total usable: %d  |  Skipped (no hand): %d", len(X), skipped)

    if len(X) == 0:
        logger.error("No landmarks could be extracted. Check your images.")
        sys.exit(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), class_names


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(num_classes: int):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_SIZE,)),   # 126 features (both hands)
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ], name="sign_landmark_mlp_both_hands")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== Sign Language Landmark Training ===")
    logger.info("Image data: %s", IMAGE_DATA_DIR)

    ensure_task_file()

    # 1. Extract landmarks
    X, y, class_names = build_dataset()

    # 2. Train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Val: %d", len(X_train), len(X_val))

    # 3. Build & train
    model = build_model(num_classes=len(class_names))
    model.summary()

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, verbose=1
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # 4. Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Final validation accuracy: %.2f%%", val_acc * 100)

    # 5. Save artefacts
    model.save(MODEL_OUT)
    logger.info("Model saved → %s", MODEL_OUT)

    with open(CLASSES_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(class_names) + "\n")
    logger.info("Classes saved → %s", CLASSES_OUT)

    logger.info("Done!")


if __name__ == "__main__":
    main()
