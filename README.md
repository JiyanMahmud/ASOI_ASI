# ASOI: Anomaly Separation and Overlap Index

> An internal evaluation metric for unsupervised anomaly detection — no ground truth required.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-read-green.svg)](#citation)

---

## Overview

Evaluating unsupervised anomaly detection is hard — you have no labels to measure against. ASOI solves this by measuring *how well a model separates its own predictions* using two complementary signals:

| Component | Symbol | Measures |
|-----------|--------|----------|
| Isolation Index | **S** | Average distance of detected anomalies from the normal centroid (Eq. 8) |
| Hellinger Distance | **H** | Distributional overlap between detected normal and anomaly sets (Eq. 9) |

These are combined into a single score:

```
ASOI = α · S_norm + β · H        (α = 0.5314,  β = 0.4686)
```

The weights were derived empirically via mutual information across 9,300 experiments on 33 datasets.

A higher ASOI means the model's predictions are well-separated — which consistently correlates with a higher F1 score even without labels (Spearman > 0.90 on most datasets).

---

## Key Results

ASOI outperforms all classical internal clustering metrics (Silhouette, Dunn, Davies-Bouldin, Calinski-Harabasz) in correlation with the F1 score, and runs faster than Silhouette and Dunn on high-dimensional data.

> Experiments ran across 33 benchmark datasets (cybersecurity, medical, IoT, fraud) using Isolation Forest, LOF, and One-Class SVM with 100 random hyperparameter configurations each.

---

## Repository Structure

```
ASOI-Anomaly-Detection/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── asoi/
│   ├── __init__.py
│   └── metrics.py              ← Core metric: ASI, ASOI, Hellinger, Isolation Index
│
├── notebooks/
│   ├── 01_asoi_implementation.ipynb        ← Metric formulas + synthetic demo
│   ├── 02_degradation_test.ipynb           ← Controlled precision degradation test
│   ├── 03_comparison_with_baselines.ipynb  ← vs. Silhouette, Dunn, MV/EM, etc.
│   ├── 04_hyperparameter_tuning.ipynb      ← GridSearch / Optuna with ASOI scorer
│   ├── 05_sensitivity_analysis.ipynb       ← Weight sensitivity (α, β)
│   └── 06_deep_models.ipynb               ← DeepSVDD + DAGMM experiments
│
├── results/
│   ├── degradation/
│   ├── comparison/
│   ├── hyperparameter_tuning/
│   └── figures/
│
├── data/
│   └── README.md               ← Dataset sources and download instructions
│
└── paper/
    └── ASOI_paper.pdf
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/ASOI-Anomaly-Detection.git
cd ASOI-Anomaly-Detection
pip install -r requirements.txt
```

### Basic Usage

```python
from asoi.metrics import compute_asoi_from_predictions

# y_pred: 1 = normal,  -1 = anomaly
score = compute_asoi_from_predictions(X, y_pred)
print(f"ASOI: {score:.4f}")   # higher = better separation
```

### Use with any scikit-learn model

```python
from asoi.metrics import compute_asoi_for_model
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.1).fit(X_train)
score = compute_asoi_for_model(model, X_test)
```

### Use as a GridSearchCV scorer (no labels needed)

```python
from sklearn.model_selection import GridSearchCV
from asoi.metrics import asoi_scorer

grid = GridSearchCV(
    IsolationForest(),
    param_grid={'n_estimators': [50, 100, 200], 'contamination': [0.05, 0.1, 0.2]},
    scoring=asoi_scorer   # <-- drop-in, label-free
)
grid.fit(X_train)
```

### Use with Optuna (Bayesian optimisation)

```python
import optuna
from asoi.metrics import compute_asoi_for_model

def objective(trial):
    n_est = trial.suggest_int('n_estimators', 50, 500)
    cont  = trial.suggest_float('contamination', 0.01, 0.3)
    model = IsolationForest(n_estimators=n_est, contamination=cont).fit(X_train)
    return compute_asoi_for_model(model, X_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

---

## Metric Reference

All functions live in `asoi/metrics.py`.

### Core metrics

| Function | Description |
|----------|-------------|
| `compute_asoi(normal, anomaly)` | Full ASOI — Eq. 14 |
| `compute_asi(normal, anomaly)` | Anomaly Separation Index — Eq. 3 |
| `compute_asoi_from_predictions(X, y_pred)` | ASOI from a 1/-1 prediction array |
| `compute_asoi_for_model(model, X)` | ASOI from a fitted sklearn model |
| `asoi_scorer(estimator, X, y=None)` | Drop-in sklearn scorer |

### Components

| Function | Description |
|----------|-------------|
| `compute_isolation_index(normal, anomaly)` | Separation S — Eq. 8 |
| `compute_hellinger_distance(normal, anomaly)` | Overlap H via Gaussian approx — Eq. 9 |
| `compute_hellinger_distance_histogram(normal, anomaly)` | Overlap H via Rice Rule histogram — Eq. 12 |

### Supplementary / comparative metrics

| Function | Description |
|----------|-------------|
| `compute_bhattacharyya_distance(normal, anomaly)` | Bhattacharyya distance |
| `compute_mahalanobis_distance(normal, anomaly)` | Average Mahalanobis distance |

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_asoi_implementation` | Formulas, visualisations, and demo on the synthetic dataset |
| `02_degradation_test` | Incrementally swap anomaly/normal labels and track ASOI vs F1 |
| `03_comparison_with_baselines` | Correlation analysis vs Silhouette, Dunn, MV/EM metrics |
| `04_hyperparameter_tuning` | GridSearch + Optuna using ASOI as the unsupervised objective |
| `05_sensitivity_analysis` | Effect of varying α and β weights on metric stability |
| `06_deep_models` | DeepSVDD and DAGMM with ASOI-guided tuning |

---

## Datasets

Experiments used 33 benchmark and real-world datasets spanning anomaly ratios from 0.002 to 0.48:

| Domain | Datasets |
|--------|----------|
| Cybersecurity | KDDCUP, SWaT, HAI, MODbus, WUSTL-IIoT |
| Medical | BreastCancer, Thyroid, Hepatitis, Heart, Pima |
| Fraud | CreditCard, Bank |
| IoT / Sensors | NAB, CPU, Server, SKAB, HVCM |
| General ML | ALOI, Ionosphere, Sonar, Wine, WBC, Spam, Rings, Twonorm |

Download instructions and links for each dataset are in [`data/README.md`](data/README.md).

> **Note:** Datasets are not included in this repository due to size and licensing constraints.

---

## Requirements

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
scipy>=1.7
torch>=1.10
optuna>=3.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Sanity Check

Verify the metric works correctly on synthetic data:

```bash
python asoi/metrics.py
```

Expected output:
```
Well-separated  ASOI : ~0.85   <- expect high
Overlapping     ASOI : ~0.15   <- expect low
```

---

## Limitations

- Performance is **weaker on datasets with contamination < 0.05** — very sparse anomalies reduce the reliability of the Hellinger estimate.
- The fixed weights (α, β) were derived from 33 datasets; they may not be optimal for highly domain-specific data.
- ASOI assumes anomalies are the minority class. If the minority/majority split is unclear, label encoding should be verified before use.

---

## Citation

If you use ASOI in your research, please cite:

```bibtex
@article{asoi2025,
  title   = {ASOI: Anomaly Separation and Overlap Index — an internal evaluation
             metric for unsupervised anomaly detection},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
