# Machine-Learning Assignment of Pyrenean Vegetation to Phytosociological Alliances

This repository contains the code and data pipeline used in the paper *"Machine-learning assignment of Pyrenean vegetation to phytosociological alliances"*. It implements a complete workflow for classifying vegetation relevés into phytosociological alliances using three machine learning approaches: **LightGBM**, **XGBoost**, and a **Multilayer Perceptron (MLP)**.

---

## Table of Contents

- [Machine-Learning Assignment of Pyrenean Vegetation to Phytosociological Alliances](#machine-learning-assignment-of-pyrenean-vegetation-to-phytosociological-alliances)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository Structure](#repository-structure)
  - [Data Description](#data-description)
    - [Input Data Format](#input-data-format)
    - [Species Index Dictionary](#species-index-dictionary)
    - [Class Filtering](#class-filtering)
  - [Methods](#methods)
    - [1. Data Vectorization (`releve_vectorizer.ipynb`)](#1-data-vectorization-releve_vectorizeripynb)
    - [2. Model Training (LightGBM / XGBoost / MLP)](#2-model-training-lightgbm--xgboost--mlp)
    - [MLP-Specific: Autoencoder Dimensionality Reduction](#mlp-specific-autoencoder-dimensionality-reduction)
    - [Hyperparameter Spaces](#hyperparameter-spaces)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Analysis](#running-the-analysis)
  - [Workflow](#workflow)
  - [Output Files](#output-files)
  - [Reproducibility](#reproducibility)
  - [License](#license)
    - [Code](#code)
    - [Data](#data)

---

## Overview

Phytosociological classification assigns vegetation plots (relevés) to hierarchical syntaxonomic units based on their floristic composition. This process is traditionally performed by expert botanists through manual comparison with reference descriptions. In this work, we explore the use of supervised machine learning to automate the assignment of Pyrenean vegetation relevés to phytosociological alliances.

Each relevé is represented as a numerical vector encoding the abundance of 3,700 plant species. Three classification models are trained and evaluated:

| Model | Description |
|-------|-------------|
| **LightGBM** | Gradient boosting framework optimized for speed and efficiency |
| **XGBoost** | Extreme gradient boosting with regularization |
| **MLP** | Deep neural network with autoencoder-based dimensionality reduction |

All models use **Optuna** for Bayesian hyperparameter optimization with 5-fold stratified cross-validation.

---

## Repository Structure

```
pyrenean-vegetation-ml-alliance-assignment/
├── scripts/
│   ├── requirements.txt                         # Python dependencies
│   ├── releve_vectorizer.ipynb                  # Data preparation: raw relevés → feature vectors
│   ├── LightGBM_pirineus.ipynb                  # LightGBM training and evaluation
│   ├── XGBoost_Pirineus.ipynb                   # XGBoost training and evaluation
│   └── Multilayer_Perceptron_pirineus.ipynb     # MLP training and evaluation
├── training_images/
│   ├── learning_curves_xgbost_reduced.png
│   ├── lgbm_training_best_params_reduced.jpg
│   ├── lgbm_training_best_params_sparse.png
│   ├── mlp_sparse_train.png
│   ├── mlp_train_best_params_reduced_03_29_09_2024.jpg
│   └── xgboost_sparse_learning_curves.png
├── .gitignore
├── README.md
└── data.zip                                     # Input data archive
```

---

## Data Description

### Input Data Format

The primary input file (`output_pirineus_3_corrected.txt`) is a JSON dictionary with the following structure:

```json
{
  "releve_id": [[abundance_vector], ["alliance_label"]],
  ...
}
```

- **Key**: Unique relevé identifier (e.g., `"P-P00123"`)
- **Value**: A list containing:
  - A numerical vector of length 3,700 representing species abundances (values 0–9, following the Braun-Blanquet abundance-dominance scale transformed to ordinal values)
  - The assigned phytosociological alliance label (e.g., `["L345"]`)

### Species Index Dictionary

The file `sp_index_dic.txt` maps each species identifier to its position in the feature vector, ensuring consistent encoding across all relevés.

### Class Filtering

To ensure meaningful classification, only alliances represented by at least **20 relevés** in the dataset are retained for training. This threshold balances having sufficient training examples per class against maintaining broad taxonomic coverage.

---

## Methods

### 1. Data Vectorization (`releve_vectorizer.ipynb`)

Raw relevé data (species lists with abundance indices per plot) is transformed into fixed-length numerical vectors:

1. Load species occurrence records for each relevé
2. Map species to fixed positions using the species index dictionary
3. Encode abundance values at the corresponding positions
4. Output: sparse vectors of dimension = total number of species in the reference flora

### 2. Model Training (LightGBM / XGBoost / MLP)

All three training notebooks follow a consistent pipeline:

1. **Data loading** — Load pre-vectorized relevé data
2. **Preprocessing** — Encode target labels, filter rare classes
3. **Train/test split** — 70/30 stratified split (random state = 8)
4. **Hyperparameter optimization** — Optuna with 50 trials, 5-fold stratified CV
5. **Model training** — Train with best parameters
6. **Evaluation** — Learning curves, cross-validation scores, classification reports
7. **Prediction export** — Top-3 predictions with probabilities for each relevé
8. **Final model** — Retrain on 100% of data and save for deployment
9. **Inference** — Apply to unlabeled relevés

### MLP-Specific: Autoencoder Dimensionality Reduction

The MLP notebook includes an additional step where an autoencoder compresses the 3,700-dimensional input into a 500-dimensional representation before classification. This helps the neural network learn more efficiently from the sparse, high-dimensional input.

### Hyperparameter Spaces

| Model | Key Hyperparameters |
|-------|-------------------|
| LightGBM | learning_rate, num_leaves, max_depth, min_data_in_leaf, feature_fraction, bagging_fraction, L1/L2 regularization, num_iterations |
| XGBoost | learning_rate, max_depth, min_child_weight, gamma, subsample, colsample_bytree, L1/L2 regularization, n_estimators |
| MLP | n_layers, n_units, activation function, optimizer, learning_rate, epochs |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster TensorFlow training

### Installation

1. Clone this repository:
   ```bash
      gh repo clone rauletepawa/pyrenean-vegetation-ml-alliance-assignment
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # or
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

Execute the notebooks in the following order:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `releve_vectorizer.ipynb` | Vectorize raw relevé data (only needed if starting from raw data) |
| 2a | `LightGBM_pirineus.ipynb` | Train and evaluate LightGBM model |
| 2b | `XGBoost_Pirineus.ipynb` | Train and evaluate XGBoost model |
| 2c | `Multilayer_Perceptron_pirineus.ipynb` | Train and evaluate MLP model |

Steps 2a, 2b, and 2c are independent of each other and can be run in any order.

You can run the notebooks interactively in Jupyter:
```bash
jupyter notebook
```

Or execute them from the command line:
```bash
jupyter nbconvert --to notebook --execute LightGBM_pirineus.ipynb
```

> **Note**: Hyperparameter optimization (Optuna) may take significant time depending on hardware. Each study runs 50 trials with 5-fold cross-validation.

---

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Raw Relevé Data                           │
│              (species occurrences per vegetation plot)           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   releve_vectorizer.ipynb                        │
│         Transform species lists → numerical feature vectors     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Vectorized Relevé Matrix                        │
│          (n_relevés × n_species sparse matrix, values 0-9)      │
└──────────┬──────────────┼───────────────────────┬───────────────┘
           │              │                       │
           ▼              ▼                       ▼
┌────────────────┐ ┌────────────────┐ ┌──────────────────────────┐
│    LightGBM    │ │    XGBoost     │ │     MLP + Autoencoder    │
│                │ │                │ │                          │
│ • Optuna (50t) │ │ • Optuna (50t) │ │ • Autoencoder (500 dim)  │
│ • 5-fold CV    │ │ • 5-fold CV    │ │ • Optuna (50t)           │
│ • Train/Eval   │ │ • Train/Eval   │ │ • 5-fold CV              │
└───────┬────────┘ └───────┬────────┘ └────────────┬─────────────┘
        │                  │                       │
        ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation & Results                         │
│  • Classification reports  • Learning curves  • Top-3 preds     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Output Files

Each training notebook produces:

| File | Description |
|------|-------------|
| `cls_report_test_<model>.xlsx` | Per-class precision, recall, and F1-score on the test set |
| `predictions_test_<model>.xlsx` | Detailed predictions with relevé IDs, true labels, and top-3 predicted alliances with probabilities |
| `predictions_new_data_<model>.xlsx` | Predictions for unlabeled relevés |
| `models/<model>_final_model.*` | Serialized final model trained on 100% of the data |

---

## Reproducibility

- **Random state**: All data splits use `random_state=8` for reproducibility
- **Seeds**: NumPy and TensorFlow seeds are set to 42 for the autoencoder training
- **Stratification**: All train/test splits and cross-validation folds use stratified sampling to preserve class distributions

---

## License

### Code
The scripts and notebooks in this repository are released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Raúl Pérez, Gwendolyn Peyre, Xavier Font

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Data
The datasets provided in this repository are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to share and adapt the data for any purpose, provided appropriate credit is given to the original authors. See [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/) for full terms.

---




