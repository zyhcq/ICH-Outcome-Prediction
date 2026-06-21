# ICH Outcome Prediction Pipeline

This repository contains the core pipeline for an imaging-only prognostic model that predicts 90-day functional outcome after intracerebral hemorrhage (ICH).

The model requires only two inputs:
1. **Non-contrast CT scan**
2. **Patient Age**

## Overview

The proposed prognostic tool addresses limitations of existing clinical severity scores like the ICH Score. It relies on automated CT feature extraction (including novel Patient-Adaptive Stratified Habitat [PASH] metrics) and uses an imaging-derived Glasgow Coma Scale proxy (iGCS) to predict patient outcomes (mRS 3-6) with high accuracy and interpretability.

## Directory Structure

* `efficientmednext/`: Contains deep learning source code for training and running the nnU-Net/MedNeXt style segmentation. This stage masks and isolates the hematoma from the raw non-contrast CT.
* `habitat/`: Contains the traditional machine learning and statistical pipeline.
  * `scripts/`: The core pipeline scripts to generate PASH features, run cross-validation, external test evaluations, and build the 23-feature L2-regularized logistic regression model (iGCS-23).

## Key Innovations

- **Automated Hematoma Segmentation**: Uses EfficientMedNeXt-L to accurately isolate hemorrhage on CT scans without manual annotation.
- **PASH Features**: Automatically stratifies the hematoma by Hounsfield units (HU) to measure high-density fragmentation and low-density dispersion, serving as strong imaging biomarkers.
- **Imaging-derived GCS (iGCS)**: Bypasses the subjective bedside GCS by regressing automated CT features against true GCS as a surrogate target during training. At test time, iGCS provides a structurally-anchored severity surrogate without needing bedside exams.
- **High Discrimination (iGCS-23)**: The full 23-feature model was extensively validated internally and externally to significantly outperform the established ICH Score.
- **Highly Interpretable Minimal Model (iGCS + Age)**: A stripped-down, two-variable model combining only the predicted iGCS and patient age. It provides near-equivalent predictive performance to the 23-feature model while offering supreme clinical interpretability through a straightforward nomogram, allowing clinicians to transparently assess risk without a "black-box" algorithm.

## Usage

1. Start by preprocessing CT data and executing segmentation inside `efficientmednext`.
2. Extract mathematical morphology, density, and PASH features using the scripts in `habitat/scripts/`.
3. Run the evaluation and regression pipelines (e.g., `05_v5_run_pipeline.py`) to reproduce validation metrics and deploy the predictive nomogram.
