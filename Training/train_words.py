"""
Train an LSTM model for ISL word-level sign recognition.

Dataset structure:
  Frames_Word_Level/
    <WORD_LABEL>/
      1/   frame01.jpg, frame02.jpg ...
      2/   ...

Outputs:
  sign_word_model.keras     – trained LSTM model
  word_classes.txt          – word labels (one per line)

Usage:
  python train_words.py
"""

from __future__ import annotations
import os, sys, logging, warnings, urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = r"D:\archive\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"
MODEL_OUT    = os.path.join(SCRIPT_DIR, "sign_word_model.keras")
CLASSES_OUT  = os.path.join(SCRIPT_DIR, "word_classes.txt")
TASK_FILE    = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
TASK_URL     = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

SEQ_LEN      = 10    # most word classes have ~9 frames; pad/truncate to 10
FEATURE_SIZE = 126   # 2 hands × 21 landmarks × 3 (x,y,z)


def ensure_task_file():
    if os.path.isfile(TASK_FILE) and os.path.getsize(TASK_FILE) > 100_000:
        return
    logger.info("Downloading hand_landmarker.task …")
    urllib.request.urlretrieve(TASK_URL, TASK_FILE)
    logger.info("Downloaded (%.1f MB)", os.path.getsize(TASK_FILE) / 1e6)


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
        out[i*3], out[i*3+1], out[i*3+2] = xs[i], ys[i], zs[i]
    return out


def frame_to_features(img_path: str, detector) -> np.ndarray:
    import mediapipe as mp
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(FEATURE_SIZE, dtype=np.float32)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    if result.hand_landmarks:
        for lm, hand in zip(result.hand_landmarks, result.handedness):
            label = hand[0].category_name
            slot = 0 if label == "Left" else 63
            features[slot:slot+63] = normalize_hand(lm)
    return features


def build_dataset():
    """
    Dataset layout: Frames_Word_Level/<WORD>/<img1>.jpg, <img2>.jpg ...
    Each class folder contains images directly (flat layout, no recording subfolders).
    All images in a class folder form a single sequence for that word.
    """
    word_names = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not word_names:
        logger.error("No word folders found in %s", DATASET_DIR)
        sys.exit(1)

    logger.info("Found %d word classes", len(word_names))
    X, y = [], []
    detector = make_detector()

    for label_idx, word in enumerate(word_names):
        word_dir = os.path.join(DATASET_DIR, word)

        # Support both flat (images directly in word_dir) and nested (recording subfolders)
        subdirs = sorted(
            d for d in os.listdir(word_dir)
            if os.path.isdir(os.path.join(word_dir, d))
        )

        if subdirs:
            # Nested layout: each subdir is one recording
            sequences = []
            for rec in subdirs:
                rec_dir = os.path.join(word_dir, rec)
                frames = sorted(
                    f for f in os.listdir(rec_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    and f.lower() != 'thumbs.db'
                )
                if frames:
                    sequences.append((rec_dir, frames))
        else:
            # Flat layout: all images form one sequence
            frames = sorted(
                f for f in os.listdir(word_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                and f.lower() != 'thumbs.db'
            )
            sequences = [(word_dir, frames)] if frames else []

        count = 0
        for seq_dir, frame_list in sequences:
            seq = []
            for fname in frame_list:
                feat = frame_to_features(os.path.join(seq_dir, fname), detector)
                seq.append(feat)

            if not seq:
                continue

            seq = np.array(seq, dtype=np.float32)
            if len(seq) >= SEQ_LEN:
                indices = np.linspace(0, len(seq)-1, SEQ_LEN, dtype=int)
                seq = seq[indices]
            else:
                pad = np.zeros((SEQ_LEN - len(seq), FEATURE_SIZE), dtype=np.float32)
                seq = np.vstack([seq, pad])

            X.append(seq)
            y.append(label_idx)
            count += 1

        logger.info("  %-30s  %d sequence(s)", word[:30], count)

    detector.close()

    if not X:
        logger.error("No sequences extracted.")
        sys.exit(1)

    logger.info("Total sequences: %d", len(X))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), word_names


def build_model(num_classes: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, FEATURE_SIZE)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name="isl_word_lstm")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    logger.info("=== ISL Word LSTM Training ===")
    ensure_task_file()

    X, y, class_names = build_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(np.bincount(y)) >= 2 else None
    )
    logger.info("Train: %d  Val: %d", len(X_train), len(X_val))

    import tensorflow as tf
    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ]

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=60, batch_size=16,
              callbacks=callbacks, verbose=1)

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Final validation accuracy: %.2f%%", val_acc * 100)

    model.save(MODEL_OUT)
    logger.info("Model saved → %s", MODEL_OUT)

    with open(CLASSES_OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names) + '\n')
    logger.info("Classes saved → %s  (%d words)", CLASSES_OUT, len(class_names))
    logger.info("Done!")


if __name__ == "__main__":
    main()
