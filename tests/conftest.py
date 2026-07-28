"""
Test configuration for sign-language-recognition API tests.

Creates a minimal dummy model and classes.txt before tests run so that
the FastAPI lifespan can start successfully without a real trained model.
The artefacts are removed after the test session.
"""

from __future__ import annotations

import os
import shutil
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pytest

# Paths must match what main.py expects (relative to d:\sign\)
MODEL_PATH = "sign_image_model.h5"
CLASSES_PATH = "classes.txt"

# 36 ASL classes: A-Z + 0-9
CLASS_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]


def _build_dummy_model(num_classes: int = 36):
    """Build and return a minimal (untrained) Keras model with the correct I/O shape."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf  # noqa: F401

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(4, 3, activation="relu", padding="same"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@pytest.fixture(scope="session", autouse=True)
def dummy_model_artifacts(tmp_path_factory):
    """
    Session-scoped fixture that creates sign_image_model.h5 and classes.txt
    in the project root before any test runs and removes them afterwards.

    Works whether or not the files already exist (backs up existing files).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dest = os.path.join(root, MODEL_PATH)
    classes_dest = os.path.join(root, CLASSES_PATH)

    # ------------------------------------------------------------------
    # Back up any existing artefacts so we can restore them after tests
    # ------------------------------------------------------------------
    backed_up_model = None
    backed_up_classes = None

    if os.path.isfile(model_dest):
        backed_up_model = model_dest + ".bak"
        shutil.move(model_dest, backed_up_model)

    if os.path.isfile(classes_dest):
        backed_up_classes = classes_dest + ".bak"
        shutil.move(classes_dest, backed_up_classes)

    # ------------------------------------------------------------------
    # Create dummy artefacts
    # ------------------------------------------------------------------
    model = _build_dummy_model(num_classes=len(CLASS_LABELS))
    model.save(model_dest)

    with open(classes_dest, "w", encoding="utf-8") as fh:
        for label in CLASS_LABELS:
            fh.write(label + "\n")

    yield  # --- tests run here ---

    # ------------------------------------------------------------------
    # Clean up / restore
    # ------------------------------------------------------------------
    if os.path.isfile(model_dest):
        os.remove(model_dest)
    if os.path.isfile(classes_dest):
        os.remove(classes_dest)

    if backed_up_model and os.path.isfile(backed_up_model):
        shutil.move(backed_up_model, model_dest)
    if backed_up_classes and os.path.isfile(backed_up_classes):
        shutil.move(backed_up_classes, classes_dest)
