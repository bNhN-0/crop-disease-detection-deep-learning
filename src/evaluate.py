from __future__ import annotations

import argparse
import json
import os
import sys

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import ensure_best_model, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best model on the valid split.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/efficientnetb0_config.json",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_path = resolve_path(config["dataset_path"])
    valid_dir = os.path.join(dataset_path, config["evaluation_split"])
    class_names_path = resolve_path(config["class_names_path"])
    metrics_path = os.path.join(ROOT, "results", "best_model_metrics.json")

    os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    best_model_path = ensure_best_model(config)

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=tuple(config["image_size"]),
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    class_names = valid_ds.class_names
    if not os.path.exists(class_names_path):
        with open(class_names_path, "w", encoding="utf-8") as file:
            json.dump(class_names, file, indent=2)

    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(best_model_path)
    validation_loss, validation_accuracy = model.evaluate(valid_ds, verbose=1)

    metrics = {
        "model": config["model_name"],
        "model_path": os.path.relpath(best_model_path, ROOT).replace("\\", "/"),
        "validation_accuracy": float(validation_accuracy),
        "validation_loss": float(validation_loss),
        "evaluation_split": config["evaluation_split"],
        "num_classes": int(config["num_classes"]),
        "class_names_path": os.path.relpath(class_names_path, ROOT).replace("\\", "/"),
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Validation accuracy: {validation_accuracy:.6f}")
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
