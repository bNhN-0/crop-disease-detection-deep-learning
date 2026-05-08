from __future__ import annotations

import argparse
import os
import sys

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import get_top_predictions, load_class_names, load_image_for_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop disease class for one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", required=True, help="Path to the saved Keras model.")
    parser.add_argument("--class-names", required=True, help="Path to class names JSON.")
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def main() -> None:
    args = parse_args()

    image_path = resolve_path(args.image)
    model_path = resolve_path(args.model)
    class_names_path = resolve_path(args.class_names)

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(class_names_path)

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)
    image_batch = load_image_for_prediction(image_path, (224, 224))

    predictions = model.predict(image_batch, verbose=0)
    top_predictions = get_top_predictions(predictions, class_names, top_k=3)
    top_prediction = top_predictions[0]

    print("Top Prediction:")
    print(f"Crop: {top_prediction['crop']}")
    print(f"Condition: {top_prediction['condition']}")
    print(f"Confidence: {top_prediction['confidence'] * 100:.2f}%")
    print()
    print("Top 3 Predictions:")
    for index, item in enumerate(top_predictions, start=1):
        print(
            f"{index}. {item['crop']} | {item['condition']} | "
            f"{item['confidence'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
