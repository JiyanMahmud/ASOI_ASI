"""
ASOI: Anomaly Separation and Overlap Index
==========================================
An internal evaluation metric for unsupervised anomaly detection.

This module implements:
    - ASI  (Anomaly Separation Index)         — Eq. 3  in the paper
    - ASOI (Anomaly Separation and Overlap Index) — Eq. 14 in the paper

Components
----------
- Isolation Index (S) : average distance of anomalous points to the
  centroid of normal data, min-max normalised.           (Eq. 8 / Alg. 1 step 2-5)
- Hellinger Distance  (H) : distributional overlap between anomaly and
  normal distributions, computed per feature and averaged. (Eq. 9 / Alg. 1 step 6-8)

Weights (from paper, Section 6.2):
    alpha = 0.5314  (separation / isolation index)
    beta  = 0.4686  (Hellinger overlap)

Reference
---------
"ASOI: Anomaly Separation and Overlap Index — an internal evaluation
metric for unsupervised anomaly detection" (2025)

Usage
-----
    from metrics import compute_asoi, compute_asi, compute_hellinger_distance, compute_isolation_index

    # y_pred: array of 1 (normal) and -1 (anomaly) predictions
    score = compute_asoi(X, y_pred)
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

# ---------- Paper weights (Section 6.2, derived via mutual information) ----------
ALPHA = 0.5314   # weight for separation component (S)
BETA  = 0.4686   # weight for Hellinger overlap component (H)


# =============================================================================
# LOW-LEVEL COMPONENTS
# =============================================================================

def compute_hellinger_distance(normal_data: np.ndarray,
                                anomaly_data: np.ndarray) -> float:
    """
    Compute the average per-feature Hellinger distance between the
    normal and anomaly distributions using Gaussian approximation.

    This is the overlap component H in ASOI (Eq. 9 of the paper).
    The Hellinger distance ranges in [0, 1]:
        H = 0  → identical distributions (maximum overlap)
        H = 1  → completely disjoint distributions (no overlap)

    Implementation note
    -------------------
    The exact PDFs are unavailable so they are approximated per feature
    by fitting a Gaussian (mean, std).  Numerical integration is then
    used to compute the Bhattacharyya coefficient, from which H follows.
    The paper alternatively describes a histogram-based approach (Rice
    Rule bins); the Gaussian approximation used here is equivalent in
    expectation and is faster for moderate dimensionalities.

    Parameters
    ----------
    normal_data  : np.ndarray, shape (n_normal,  d)
    anomaly_data : np.ndarray, shape (n_anomaly, d)

    Returns
    -------
    float  Average Hellinger distance across all d features.
    """
    normal_data  = np.asarray(normal_data,  dtype=np.float64)
    anomaly_data = np.asarray(anomaly_data, dtype=np.float64)

    if normal_data.ndim  == 1: normal_data  = normal_data.reshape(-1,  1)
    if anomaly_data.ndim == 1: anomaly_data = anomaly_data.reshape(-1, 1)

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return 0.0

    distances = []

    for i in range(normal_data.shape[1]):
        mean_n, std_n = np.mean(normal_data[:, i]),  np.std(normal_data[:, i])
        mean_a, std_a = np.mean(anomaly_data[:, i]), np.std(anomaly_data[:, i])

        # Prevent degenerate distributions
        std_n = max(std_n, 1e-8)
        std_a = max(std_a, 1e-8)

        # Integration range with 20 % buffer on each side
        min_val   = min(np.min(normal_data[:, i]),  np.min(anomaly_data[:, i]))
        max_val   = max(np.max(normal_data[:, i]),  np.max(anomaly_data[:, i]))
        span      = max_val - min_val
        buffer    = span * 0.2 if span > 1e-8 else 1.0
        x         = np.linspace(min_val - buffer, max_val + buffer, 1000)

        px = np.maximum(norm.pdf(x, mean_n, std_n), 0)
        qx = np.maximum(norm.pdf(x, mean_a, std_a), 0)

        # Bhattacharyya coefficient  BC = ∫ √(p·q) dx
        bc = np.clip(np.trapz(np.sqrt(px * qx), x), 0, 1)

        # Hellinger distance  H = √(1 − BC)
        distances.append(np.sqrt(1.0 - bc))

    return float(np.mean(distances)) if distances else 0.0


def compute_isolation_index(normal_data: np.ndarray,
                             anomaly_data: np.ndarray,
                             normalise: bool = True) -> float:
    """
    Compute the Isolation Index S (Eq. 8 of the paper).

    S is the mean Euclidean distance from each anomalous point to the
    centroid of the normal data — measuring how far anomalies deviate
    from the compact normal cluster.

    When ``normalise=True`` (default) S is divided by the length of the
    diagonal of the joint bounding box, placing S in [0, 1].
    This corresponds to steps 2–5 of Algorithm 1 in the paper.

    Parameters
    ----------
    normal_data  : np.ndarray, shape (n_normal,  d)
    anomaly_data : np.ndarray, shape (n_anomaly, d)
    normalise    : bool  Whether to min-max normalise the index.

    Returns
    -------
    float  (Normalised) isolation index.
    """
    normal_data  = np.asarray(normal_data,  dtype=np.float64)
    anomaly_data = np.asarray(anomaly_data, dtype=np.float64)

    if normal_data.ndim  == 1: normal_data  = normal_data.reshape(-1,  1)
    if anomaly_data.ndim == 1: anomaly_data = anomaly_data.reshape(-1, 1)

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return 0.0

    # Centroid of normal data  cN
    centroid_normal = np.mean(normal_data, axis=0)

    # S = (1/|A|) Σ ‖x_a − c_N‖
    distances       = np.sqrt(np.sum((anomaly_data - centroid_normal) ** 2, axis=1))
    isolation_index = float(np.mean(distances))

    if normalise:
        # Max possible distance = diagonal of joint bounding box
        overall_min = np.minimum(np.min(normal_data,  axis=0),
                                 np.min(anomaly_data, axis=0))
        overall_max = np.maximum(np.max(normal_data,  axis=0),
                                 np.max(anomaly_data, axis=0))
        max_dist    = np.linalg.norm(overall_max - overall_min)

        isolation_index = isolation_index / max_dist if max_dist > 1e-8 else 0.0

    return isolation_index


# =============================================================================
# HIGH-LEVEL METRIC FUNCTIONS
# =============================================================================

def compute_asi(normal_data: np.ndarray,
                anomaly_data: np.ndarray) -> float:
    """
    Anomaly Separation Index (ASI) — Eq. 3 of the paper.

    ASI = ‖μ_N − μ_A‖₂  /  √(Σ_j σ²_pooled,j)

    A Cohen's-d-style effect size adapted to high-dimensional data.
    Higher → greater separation between normal and anomalous distributions.

    Parameters
    ----------
    normal_data  : np.ndarray, shape (n_normal,  d)
    anomaly_data : np.ndarray, shape (n_anomaly, d)

    Returns
    -------
    float  ASI value ≥ 0.
    """
    normal_data  = np.asarray(normal_data,  dtype=np.float64)
    anomaly_data = np.asarray(anomaly_data, dtype=np.float64)

    if normal_data.ndim  == 1: normal_data  = normal_data.reshape(-1,  1)
    if anomaly_data.ndim == 1: anomaly_data = anomaly_data.reshape(-1, 1)

    if normal_data.shape[0] < 2 or anomaly_data.shape[0] < 2:
        return 0.0

    n_n = normal_data.shape[0]
    n_a = anomaly_data.shape[0]

    mu_n = np.mean(normal_data,  axis=0)   # μ_N  (Eq. 4)
    mu_a = np.mean(anomaly_data, axis=0)   # μ_A

    var_n = np.var(normal_data,  axis=0, ddof=1)   # σ²_{N,j}  (Eq. 6)
    var_a = np.var(anomaly_data, axis=0, ddof=1)   # σ²_{A,j}  (Eq. 7)

    # Pooled variance per feature  (Eq. 5)
    pooled_var = ((n_n - 1) * var_n + (n_a - 1) * var_a) / (n_n + n_a - 2)
    pooled_var = np.maximum(pooled_var, 1e-16)   # prevent division by zero

    numerator   = np.linalg.norm(mu_n - mu_a)          # ‖μ_N − μ_A‖₂
    denominator = np.sqrt(np.sum(pooled_var))           # √(Σ σ²_pooled,j)

    return float(numerator / denominator) if denominator > 1e-16 else 0.0


def compute_asoi(normal_data: np.ndarray,
                 anomaly_data: np.ndarray,
                 alpha: float = ALPHA,
                 beta: float  = BETA) -> float:
    """
    Anomaly Separation and Overlap Index (ASOI) — Eq. 14 / Algorithm 1.

    ASOI = α · S_norm + β · H

    where S_norm is the normalised isolation index (separation component)
    and H is the average Hellinger distance (overlap component).

    Higher ASOI → better anomaly detection performance (well-separated,
    low-overlap predictions).

    Parameters
    ----------
    normal_data  : np.ndarray, shape (n_normal,  d)
    anomaly_data : np.ndarray, shape (n_anomaly, d)
    alpha        : float  Weight for the separation component (default 0.5314).
    beta         : float  Weight for the overlap component    (default 0.4686).

    Returns
    -------
    float  ASOI ∈ [0, 1].
    """
    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return 0.0

    S = compute_isolation_index(normal_data, anomaly_data, normalise=True)
    H = compute_hellinger_distance(normal_data, anomaly_data)

    score = alpha * S + beta * H
    return float(score) if np.isfinite(score) else 0.0


def compute_asoi_from_predictions(X: np.ndarray,
                                   y_pred: np.ndarray,
                                   alpha: float = ALPHA,
                                   beta: float  = BETA) -> float:
    """
    Convenience wrapper: compute ASOI directly from a prediction array.

    Parameters
    ----------
    X      : np.ndarray, shape (n_samples, d)   Feature matrix.
    y_pred : np.ndarray, shape (n_samples,)      Predictions — 1 = normal, -1 = anomaly.
    alpha  : float  Weight for separation component  (default 0.5314).
    beta   : float  Weight for Hellinger component   (default 0.4686).

    Returns
    -------
    float  ASOI ∈ [0, 1], or -inf if one class has no predicted samples.
    """
    X      = np.asarray(X)
    y_pred = np.asarray(y_pred)

    normal_mask  = y_pred == 1
    normal_data  = X[ normal_mask]
    anomaly_data = X[~normal_mask]

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return -np.inf   # degenerate prediction — all one class

    return compute_asoi(normal_data, anomaly_data, alpha=alpha, beta=beta)


def compute_asoi_for_model(model, X, alpha: float = ALPHA, beta: float = BETA) -> float:
    """
    Compute ASOI for a fitted scikit-learn-compatible anomaly detector.

    Calls ``model.predict(X)`` (which must return 1 / -1) and then
    evaluates ASOI on the resulting split of X.

    Parameters
    ----------
    model  : fitted estimator with a .predict() method.
    X      : array-like, shape (n_samples, d).
    alpha  : float  Weight for separation (default 0.5314).
    beta   : float  Weight for Hellinger overlap (default 0.4686).

    Returns
    -------
    float  ASOI score, or -inf on error / degenerate output.
    """
    try:
        X_np   = X.values if hasattr(X, 'values') else np.asarray(X)
        y_pred = model.predict(X_np)
        return compute_asoi_from_predictions(X_np, y_pred, alpha=alpha, beta=beta)
    except Exception as exc:
        print(f"[ASOI] Error computing score: {exc}")
        return -np.inf


# =============================================================================
# SCIKIT-LEARN COMPATIBLE SCORER
# =============================================================================

def asoi_scorer(estimator, X, y=None):
    """
    Drop-in scorer for GridSearchCV / RandomizedSearchCV / Optuna wrappers.

    Because ASOI is label-free, y is accepted but ignored — making this
    compatible with sklearn's scoring API while remaining purely unsupervised.

    Pass to GridSearchCV via:
        from sklearn.metrics import make_scorer
        scoring = make_scorer(asoi_scorer, needs_proba=False, needs_threshold=False)
    Or directly:
        GridSearchCV(estimator, param_grid, scoring=asoi_scorer)

    Returns
    -------
    float  ASOI score (higher is better), or -inf on failure.
    """
    return compute_asoi_for_model(estimator, X)


# =============================================================================
# QUICK SANITY-CHECK (run as script)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Compact normal cluster centred at origin
    normal  = rng.normal(loc=0.0, scale=1.0,  size=(300, 5))
    # Anomalies shifted far away — should yield high ASOI
    anomaly = rng.normal(loc=5.0, scale=0.5,  size=(30,  5))
    # Anomalies overlapping with normal — should yield low ASOI
    overlap = rng.normal(loc=0.2, scale=1.0,  size=(30,  5))

    print("=== ASOI sanity check ===")
    print(f"Well-separated  ASOI : {compute_asoi(normal, anomaly):.4f}  (expect high)")
    print(f"Overlapping     ASOI : {compute_asoi(normal, overlap):.4f}  (expect low )")
    print()
    print(f"Well-separated  ASI  : {compute_asi(normal, anomaly):.4f}")
    print(f"Overlapping     ASI  : {compute_asi(normal, overlap):.4f}")
    print()

    # Test the from-predictions helper
    X_all  = np.vstack([normal, anomaly])
    y_pred = np.array([1] * 300 + [-1] * 30)
    print(f"compute_asoi_from_predictions : {compute_asoi_from_predictions(X_all, y_pred):.4f}")
