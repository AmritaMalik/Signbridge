"""
Training script for ASL sign language recognition CNN.

Loads images from Image_Data/<Label>/ directories, trains a CNN,
and saves the model as sign_image_model.h5 and class labels as classes.txt.
"""

import os
import logging
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 64


def load_dataset(root: str = "Image_Data"):
    """
    Load all images from Image_Data/<Label>/ subdirectories.

    Labels are sorted for reproducible class index assignment.
    Images are resized to 64x64, converted BGR->RGB, and normalised to [0, 1].
    Unreadable files are skipped with a warning.

    Returns:
        images (np.ndarray): shape (N, 64, 64, 3), dtype float32
        labels (np.ndarray): shape (N,), dtype int32 — integer class indices
        class_names (list[str]): sorted list of label names
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root!r}")

    class_names = sorted(
        entry.name
        for entry in os.scandir(root)
        if entry.is_dir()
    )

    if not class_names:
        raise ValueError(f"No label subdirectories found in {root!r}")

    images = []
    labels = []

    for class_idx, label in enumerate(class_names):
        label_dir = os.path.join(root, label)
        file_names = sorted(os.listdir(label_dir))
        loaded_count = 0

        for fname in file_names:
            fpath = os.path.join(label_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # Skip non-image files quickly
            lower = fname.lower()
            if not any(lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                continue

            img = cv2.imread(fpath)
            if img is None:
                logger.warning("Could not read image (skipping): %s", fpath)
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(class_idx)
            loaded_count += 1

        logger.info("Loaded %d images for class '%s'", loaded_count, label)

    if not images:
        raise ValueError("No images were loaded from the dataset.")

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), class_names


def build_model(num_classes: int = 36):
    """
    Build and compile the CNN architecture.

    Architecture:
        Input(64, 64, 3)
        -> Conv2D(32, relu, same) + MaxPool
        -> Conv2D(64, relu, same) + MaxPool
        -> Conv2D(128, relu, same) + MaxPool
        -> Flatten
        -> Dense(256, relu)
        -> Dropout(0.5)
        -> Dense(num_classes, softmax)

    Compiled with adam + sparse_categorical_crossentropy.
    """
    import tensorflow as tf  # deferred import to avoid slow startup at module level

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            # Block 1
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            # Block 2
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            # Block 3
            tf.keras.layers.Conv2D(128, kernel_size=3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            # Classifier head
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_artifacts(model, class_names: list, model_path: str = "sign_image_model.h5", classes_path: str = "classes.txt"):
    """
    Save the trained Keras model and the ordered class labels.

    Args:
        model: trained tf.keras.Model
        class_names: ordered list of label strings
        model_path: destination for the .h5 model file
        classes_path: destination for the classes text file
    """
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    with open(classes_path, "w", encoding="utf-8") as f:
        for label in class_names:
            f.write(label + "\n")
    logger.info("Class labels saved to %s (%d classes)", classes_path, len(class_names))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split  # optional dep; add to requirements if needed

    logger.info("Loading dataset from Image_Data/ ...")
    X, y, class_names = load_dataset("Image_Data")
    logger.info("Dataset loaded: %d samples, %d classes", len(X), len(class_names))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(num_classes=len(class_names))
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=20,
        validation_data=(X_val, y_val),
        batch_size=32,
    )

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nFinal training accuracy:   {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    save_artifacts(model, class_names)
