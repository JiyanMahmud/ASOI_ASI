

# ASOI:anomaly separation and overlap index,an internalevaluationmetricfor unsupervised anomaly detection

This repository contains the reference implementation and experiments for **ASOI** (Anomaly Separation and Overlap Index) and **ASI** (Anomaly Separation Index), two intrinsic, label-free metrics for evaluating unsupervised anomaly detection (AD).
The code provides metrics, experiment pipelines, and notebooks to reproduce the study’s results across synthetic and real-world datasets.

## Why ASOI / ASI?

* **Label-free:** Do not require ground-truth labels; suitable when labels are scarce, noisy, or unavailable.
* **Interpretability:** ASI quantifies separation; ASOI augments this with distributional overlap, improving robustness.
* **Correlates with supervised metrics:** Empirically aligns with F1 and related measures across algorithms and datasets.
* **Efficient & scalable:** Works with common AD workflows and large feature spaces.

## Repository structure

```
.
├─ notebooks/                 # Jupyter notebooks (exploration, figures, ablations)
├─ asoi/                      # Python package (metrics, utilities)
│  ├─ __init__.py
│  ├─ metrics.py              # ASI / ASOI implementations
│  ├─ utils.py                # helpers (preprocessing, scoring, IO)
│  └─ evaluation.py           # correlation, degradation tests, reporting
├─ experiments/               # Scripts to reproduce paper experiments
├─ data/                      #  datasets 
├─ results/                   # outputs, tables, logs
├─ figures/                   # generated plots
├─ requirements.txt           # Python dependencies
└─ README.md
```




**Distance choice:** We use **Hellinger distance** by default due to its distribution-agnostic applicability to both parametric and nonparametric families and its boundedness on ([0,1]); larger values between the marginal laws of (N) and (A) indicate greater separability.

## Reproducing the paper’s results

* **Precision-degradation tests:** See `experiments/run_precision_degradation.py`.
* **Algorithm sweep (e.g., Isolation Forest, LOF, OCSVM):** See `experiments/run_algorithms.py`.
* **Correlation with supervised metrics (F1, Spearman):** See `notebooks/03_correlation_analysis.ipynb`.
* **weight sensitivity:** See `notebooks/04_ablation_studies.ipynb`.


## Datasets

The code supports heterogeneous datasets (synthetic and real-world). They are available publicly. 


## Citation

If you use ASOI/ASI or this codebase, please cite the paper:

```bibtex

```



For questions or collaboration, open an issue or contact **jiyan@inf.elte.hu**.

---

