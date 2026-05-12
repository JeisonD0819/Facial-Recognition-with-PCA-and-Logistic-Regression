# Facial Recognition with PCA and Logistic Regression

**Universidad Nacional de Colombia**

> Miguel Antonio Parrado Pardo · Jeison Nicolás Díaz Arciniegas · David Santiago Nagles Barajas

---

## Overview

This project explores facial image processing, dimensionality reduction, and binary classification using classical machine learning techniques. Working with a dataset of 45 face images from three individuals — Bruce Lee, Neil Patrick Harris, and Pam Grier — the notebook covers the full pipeline: data loading, image transformations, PCA analysis, and logistic regression training.

---

## Objectives

- Load and preprocess a facial image dataset using PyTorch and MTCNN face detection.
- Apply image transformations (grayscale, black-and-white, Gaussian blur) and compare their effects.
- Perform Principal Component Analysis (PCA) to reduce dimensionality and visualize class separation.
- Train a logistic regression classifier to detect Bruce Lee images in a binary classification setting.
- Analyze the effect of Gaussian noise on model bias and variance.

---

## Project Structure

```
├── code/
│   └── Homework_2.ipynb       # Main notebook with all sections
├── report/
│   └── ReportPCALogisticRegression.pdf                    # Project report
└── README.md
```

> The `faces/` dataset directory is downloaded automatically when running the notebook.

---

## Dependencies

Install all required packages before running the notebook:

```bash
pip install numpy==1.26.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch --upgrade
pip install scikit-learn matplotlib Pillow
```

> **Note:** A GPU runtime is recommended for faster face detection with MTCNN, but the notebook runs on CPU as well.

---

## Sections

### Section 1 — Data Loading & Transformations

- The dataset is downloaded automatically from a public OSF repository.
- MTCNN detects and crops faces from each image.
- Three transformations are applied and visualized:
  - **Grayscale** — single-channel luminance conversion
  - **Black & White** — binary thresholding at 0.5
  - **Gaussian Blur** — spatial smoothing with `kernel_size=15`

### Section 2 — PCA Analysis

- PCA is applied to the full dataset (45 images) for each transformation.
- Outputs include:
  - 2D scatter plot of projections colored by person
  - Cumulative explained variance curve
  - Per-image PCA reconstruction using 1, 2, and 3 components
- **Key finding:** The original RGB dataset produces the clearest class separation in PCA space, since color encodes discriminative facial features that are lost after filtering or binarization.

### Section 3 — Classifier

- A binary label is created: Bruce Lee (1) vs. others (0).
- A **Logistic Regression** model (implemented with `nn.Linear` + sigmoid) is trained using BCE loss and SGD.
- The model is also trained on a **noise-corrupted** version of the dataset (Gaussian noise, σ = 0.2).
- **Key finding:** Noise degrades class separability, increases variance in predictions, and slightly raises bias, confirming the bias-variance tradeoff in the presence of data corruption.

---

## Results Summary

| Dataset Variant | PCA Class Separation | Classifier Behavior |
|---|---|---|
| Original RGB | ✅ Best | Stable, converges cleanly |
| Grayscale | ⚠️ Moderate | Reasonable grouping |
| Gaussian Blur | ⚠️ Overlapping | Loss of fine details |
| Black & White | ❌ Poor | Heavily overlapping clusters |
| RGB + Noise | — | Higher variance, less stable |

---

## Reproducibility

A fixed random seed is set at the beginning of the notebook:

```python
SEED = 2021
set_seed(seed=SEED)
```

This ensures consistent results across runs for NumPy, Python's `random`, and PyTorch.

---

## References

- Dataset source: [OSF Repository](https://osf.io/2kyfb/download) (originally from [ben-heil/cis_522_data](https://github.com/ben-heil/cis_522_data))
- Face detection: [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- PCA reference: [PyTorch randomness docs](https://pytorch.org/docs/stable/notes/randomness.html)
