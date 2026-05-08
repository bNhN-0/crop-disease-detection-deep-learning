from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import ensure_best_model, get_top_predictions, load_class_names, load_config, load_image_for_prediction

CONFIG_PATH = os.path.join(ROOT, "configs", "efficientnetb0_config.json")
IMAGE_SIZE = (224, 224)


def get_confidence_label(confidence: float) -> str:
    percent = confidence * 100
    if percent >= 85:
        return "High confidence"
    if percent >= 60:
        return "Medium confidence"
    return "Low confidence"


@st.cache_resource
def load_model():
    config = load_config(CONFIG_PATH)
    best_model_path = ensure_best_model(config)
    return tf.keras.models.load_model(best_model_path)


@st.cache_data
def load_names():
    config = load_config(CONFIG_PATH)
    class_names_path = os.path.join(ROOT, config["class_names_path"])
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(
            "results/class_names.json is missing. Run notebook evaluation or "
            "`python src/evaluate.py --config configs/efficientnetb0_config.json` first."
        )
    return load_class_names(class_names_path)


st.set_page_config(page_title="Crop Disease Detection", layout="centered")
st.title("Crop Disease Detection")
st.caption("Transfer learning model for crop leaf disease classification.")

try:
    model = load_model()
    class_names = load_names()
except Exception as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    image_batch = load_image_for_prediction(uploaded_file, IMAGE_SIZE)
    predictions = model.predict(image_batch, verbose=0)
    top_predictions = get_top_predictions(predictions, class_names, top_k=3)
    top_prediction = top_predictions[0]

    st.subheader("Prediction")
    st.write(f"Crop: {top_prediction['crop']}")
    st.write(f"Condition: {top_prediction['condition']}")
    st.write(f"Confidence: {top_prediction['confidence'] * 100:.2f}%")
    st.write(get_confidence_label(top_prediction["confidence"]))

    top_predictions_df = pd.DataFrame(
        [
            {
                "Crop": item["crop"],
                "Condition": item["condition"],
                "Confidence (%)": round(item["confidence"] * 100, 2),
            }
            for item in top_predictions
        ]
    )

    st.subheader("Top 3 Predictions")
    st.table(top_predictions_df)

st.caption(
    "This model predicts crop disease class from visible leaf symptoms. "
)
