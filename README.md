# Crop Disease Detection With Deep Learning

## Overview

This project focuses on crop leaf disease classification using deep learning and transfer learning.

The goal is to classify a leaf image into its crop type and disease condition. The project compares two pretrained CNN architectures:

- `MobileNetV2`
- `EfficientNetB0`

The final selected model is `EfficientNetB0`, which is used for prediction through a lightweight Streamlit app.

This project was built to practice and showcase:

- deep learning model training
- transfer learning
- model comparison
- validation-based evaluation
- classification reports
- confusion matrix analysis
- sample prediction visualization
- Grad-CAM explainability
- lightweight MLOps-style evaluation and prediction scripts

---

## Problem

The model takes a crop leaf image and predicts the crop and disease condition.

Example class label:

```text
Apple___Apple_scab
```

Meaning:

```text
Crop: Apple
Condition: Apple scab
```

Another example:

```text
Tomato___Late_blight
```

Meaning:

```text
Crop: Tomato
Condition: Late blight
```

This is an image classification project. The model predicts based on visible leaf symptoms and should not be treated as a final agricultural diagnosis tool.

---

## Dataset

The dataset used in this project is the **New Plant Diseases Dataset** from Kaggle.

Dataset source:

```text
Kaggle: New Plant Diseases Dataset
Author: vipoooool
```

Expected local dataset path:

```text
data/new_plant_diseases_dataset/
```

Expected folder structure:

```text
data/new_plant_diseases_dataset/
├── train/
├── valid/
└── test/
```

The dataset contains crop leaf images grouped by crop and disease condition classes.

Example classes:

```text
Apple___Apple_scab
Apple___Black_rot
Corn_(maize)___Common_rust
Potato___Early_blight
Tomato___Late_blight
Tomato___healthy
```

---

## Dataset Note

The dataset is kept local and is not uploaded to GitHub.

The provided `test/` folder in this dataset version is a small flat sample folder, not a proper 38-class labeled test split. Because of that, model comparison and final evaluation use the labeled `valid/` split.

---

## Model Approaches

## MobileNetV2

MobileNetV2 is used as the lightweight transfer learning baseline.

Model file:

```text
models/crop_disease_mobilenetv2_1.keras
```

MobileNetV2 requires its standard preprocessing:

```text
mobilenet_v2.preprocess_input
```

MobileNetV2 is useful as a fast and lightweight baseline model.

---

## EfficientNetB0

EfficientNetB0 is used as the stronger transfer learning model.

Model file:

```text
models/crop_disease_efficientnetb0_2.keras
```

Best model copy:

```text
models/best_model.keras
```

EfficientNetB0 is used as the final selected model because it achieved the strongest validation performance.

---

## Results

| Model | Validation Accuracy | Validation Loss |
|---|---:|---:|
| MobileNetV2 | ~95–96% | ~0.12 |
| EfficientNetB0 | ~97% | ~0.09 |

EfficientNetB0 achieved the best validation performance and is used as the final model.

---

## Model Comparison

The project compares MobileNetV2 and EfficientNetB0 on the same labeled validation split.

![Model Comparison Accuracy](results/figures/model_comparison_accuracy.png)

![Model Comparison Loss](results/figures/model_comparison_loss.png)

---

## Training Curves

Training curves are generated when the optional training cells are run.

### MobileNetV2

![MobileNetV2 Accuracy Curve](results/figures/mobilenetv2_accuracy_curve.png)

![MobileNetV2 Loss Curve](results/figures/mobilenetv2_loss_curve.png)

### EfficientNetB0

![EfficientNetB0 Accuracy Curve](results/figures/efficientnetb0_accuracy_curve.png)

![EfficientNetB0 Loss Curve](results/figures/efficientnetb0_loss_curve.png)

---

## Evaluation

The best model is evaluated using:

- validation accuracy
- validation loss
- classification report
- confusion matrix
- sample predictions

### Confusion Matrix

![EfficientNetB0 Confusion Matrix](results/figures/efficientnetb0_confusion_matrix.png)

### Sample Predictions

![EfficientNetB0 Sample Predictions](results/figures/efficientnetb0_sample_predictions.png)

---

## Explainability

Grad-CAM is used to visualize which image regions influenced the model prediction.

This helps inspect whether the model focuses on leaf disease regions instead of irrelevant background areas.

![EfficientNetB0 Grad-CAM Examples](results/figures/efficientnetb0_gradcam_examples.png)

---

## Notebook Workflow

The main deep learning workflow is notebook-based.

Run the notebooks in this order:

```text
1. notebooks/01_mobilenetv2.ipynb
2. notebooks/02_efficientnetb0.ipynb
3. notebooks/03_compare_models.ipynb
```

The notebooks cover:

```text
dataset loading
model training
model evaluation
model comparison
classification report
confusion matrix
sample predictions
Grad-CAM explainability
```

Training cells can be rerun, but saved models can also be loaded directly.

---

## Lightweight MLOps Layer

The project includes simple MLOps-style files for reproducible evaluation and prediction.

```text
configs/
├── mobilenetv2_config.json
└── efficientnetb0_config.json

src/
├── utils.py
├── predict.py
└── evaluate.py

app/
└── streamlit_app.py
```

## Configs

The config files store:

```text
model name
model path
dataset path
image size
batch size
seed
class names path
preprocessing requirement
```

---

## Evaluation Script

Run evaluation from the project root:

```bash
python src/evaluate.py --config configs/efficientnetb0_config.json
```

This evaluates the selected model on the labeled validation split and saves metrics.

---

## Prediction Script

Run single-image prediction:

```bash
python src/predict.py --image path/to/leaf.jpg --model models/best_model.keras --class-names results/class_names.json
```

Output includes:

```text
crop
condition
confidence
top 3 predictions
```

---

## Streamlit Demo

A simple Streamlit app is included for image upload and prediction.

Run:

```bash
streamlit run app/streamlit_app.py
```

The app shows:

```text
uploaded image
predicted crop
predicted condition
confidence score
confidence level
top 3 predictions
basic limitation note
```

The app uses:

```text
models/best_model.keras
```

---

## Outputs

Saved outputs are stored under `results/` and `results/figures/`.

Key outputs:

```text
results/best_model_classification_report.csv
results/class_names.json
results/model_comparison.json
results/model_comparison_table.csv
results/best_model_metrics.json
results/efficientnetb0_classification_report.csv
results/efficientnetb0_metrics.json
results/mobilenetv2_metrics.json
results/evaluation_metrics.json

```

Figures:

```text
results/figures/model_comparison_accuracy.png
results/figures/model_comparison_loss.png
results/figures/mobilenetv2_accuracy_curve.png
results/figures/mobilenetv2_loss_curve.png
results/figures/efficientnetb0_accuracy_curve.png
results/figures/efficientnetb0_loss_curve.png
results/figures/efficientnetb0_confusion_matrix.png
results/figures/efficientnetb0_sample_predictions.png
results/figures/efficientnetb0_gradcam_examples.png
```

---

## Repository Structure

```text
crop-disease-detection-deep-learning/
├── notebooks/
│   ├── 01_mobilenetv2.ipynb
│   ├── 02_efficientnetb0.ipynb
│   └── 03_compare_models.ipynb
│
├── configs/
│   ├── mobilenetv2_config.json
│   └── efficientnetb0_config.json
│
├── src/
│   ├── utils.py
│   ├── predict.py
│   └── evaluate.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── crop_disease_mobilenetv2_1.keras
│   ├── crop_disease_efficientnetb0_2.keras
│   └── best_model.keras
│
├── results/
│   ├── class_names.json
│   ├── model_comparison.json
│   ├── model_comparison_table.csv
│   ├── model_registry.json
│   ├── best_model_metrics.json
│   ├── efficientnetb0_classification_report.csv
│   └── figures/
│
├── data/
│   └── new_plant_diseases_dataset/
│
├── requirements.txt
└── README.md
```

---

## Limitations

- The model was trained on a PlantVillage-style dataset with mostly clean leaf images.
- Performance may drop on real-world farm images with blur, shadows, multiple leaves, or complex backgrounds.
- The model predicts based on visible leaf symptoms only.
- The output should not be treated as a final agricultural diagnosis.
- The provided `test/` folder was not used for final evaluation because it is not a full labeled 38-class split.

---