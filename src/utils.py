from __future__ import annotations

import json
import os
import shutil
from typing import Any

import numpy as np
from PIL import Image


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict[str, Any]:
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), config_path)
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_best_model(config: dict[str, Any]) -> str:
    project_root = get_project_root()
    source_model_path = config["source_model_path"]
    best_model_path = config["best_model_path"]

    if not os.path.isabs(source_model_path):
        source_model_path = os.path.join(project_root, source_model_path)
    if not os.path.isabs(best_model_path):
        best_model_path = os.path.join(project_root, best_model_path)

    if not os.path.exists(source_model_path):
        raise FileNotFoundError(source_model_path)

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    if not os.path.exists(best_model_path):
        shutil.copyfile(source_model_path, best_model_path)

    return best_model_path


def load_class_names(class_names_path: str) -> list[str]:
    if not os.path.isabs(class_names_path):
        class_names_path = os.path.join(get_project_root(), class_names_path)
    with open(class_names_path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_label(label: str) -> tuple[str, str]:
    if "___" in label:
        crop, condition = label.split("___", 1)
    else:
        crop, condition = label, "Unknown"

    crop = crop.replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()

    crop = " ".join(crop.split())
    condition = " ".join(condition.split())

    return crop, condition


def load_image_for_prediction(image: Any, image_size: tuple[int, int]) -> np.ndarray:
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    else:
        pil_image = Image.open(image).convert("RGB")

    pil_image = pil_image.resize(image_size)
    image_array = np.array(pil_image, dtype="float32")
    return np.expand_dims(image_array, axis=0)


def get_top_predictions(
    predictions: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    scores = predictions[0] if predictions.ndim > 1 else predictions
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for index in top_indices:
        label = class_names[int(index)]
        crop, condition = clean_label(label)
        confidence = float(scores[int(index)])
        results.append(
            {
                "label": label,
                "crop": crop,
                "condition": condition,
                "confidence": confidence,
            }
        )
    return results
