from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from tensorflow import keras

from src.utils import build_efficientnetb0_model, ensure_dir, load_image_datasets, load_config, save_json


def train(config_path: str | Path) -> None:
    config = load_config(config_path)
    image_size = tuple(config["image_size"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    learning_rate = float(config["learning_rate"])
    dataset_path = config["dataset_path"]
    num_classes = int(config["num_classes"])
    base_trainable = bool(config.get("base_trainable", False))

    print("Loading datasets from labeled train/valid folders...")
    train_ds, valid_ds, class_names = load_image_datasets(
        dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        seed=30,
        use_valid_as_eval=True,
    )

    model = build_efficientnetb0_model(image_size, num_classes, base_trainable=base_trainable)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model_path = ROOT / config["best_model_path"]
    final_model_path = ROOT / config["saved_model_path"]
    results_dir = ensure_dir(ROOT / "results")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(results_dir / "efficientnetb0_training_log.csv"),
    ]

    print("Training EfficientNetB0...")
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Saving final model...")
    model.save(final_model_path)

    print("Reloading best checkpoint for evaluation on the valid split...")
    best_model = keras.models.load_model(best_model_path)
    val_loss, val_accuracy = best_model.evaluate(valid_ds, verbose=1)

    metrics = {
        "model": config["model_name"],
        "training_style": "Transfer learning with frozen EfficientNetB0 base",
        "validation_accuracy": float(val_accuracy),
        "validation_loss": float(val_loss),
        "saved_model": str(Path(config["saved_model_path"]).as_posix()),
        "best_model": str(Path(config["best_model_path"]).as_posix()),
        "evaluation_split": "valid folder",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "note": "EfficientNetB0 was evaluated on the labeled validation split because the provided test folder is a tiny flat sample set.",
    }

    save_json(class_names, ROOT / "results" / "class_names.json")
    save_json(metrics, ROOT / "results" / "efficientnetb0_metrics.json")
    print(f"Saved final model to {final_model_path}")
    print(f"Saved best model to {best_model_path}")
    print("Saved class names and metrics to results/.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 for crop disease classification.")
    parser.add_argument(
        "--config",
        default="configs/efficientnetb0_config.json",
        help="Path to the training config JSON.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
