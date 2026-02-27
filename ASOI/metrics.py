"""
ASOI: Anomaly Separation and Overlap Index
==========================================
An internal evaluation metric for unsupervised anomaly detection.

This module implements the full metric family described in the paper:

    "ASOI: Anomaly Separation and Overlap Index — an internal evaluation
     metric for unsupervised anomaly detection" (2025)

Public API
----------
Core metrics (use these):
    compute_asi                          ASI  — Anomaly Separation Index (Eq. 3)
    compute_asoi                         ASOI — composite metric          (Eq. 14 / Alg. 1)
    compute_asoi_from_predictions        ASOI from a raw 1/-1 prediction array
    compute_asoi_for_model               ASOI from a fitted sklearn model
    asoi_scorer                          Drop-in sklearn GridSearchCV scorer

Component functions:
    compute_isolation_index              Separation S  (Eq. 8)
    compute_hellinger_distance           Overlap H via Gaussian approx. (Eq. 9)
    compute_hellinger_distance_histogram Overlap H via histogram bins   (Eq. 12)

Supplementary / comparative metrics:
    compute_bhattacharyya_distance       Bhattacharyya distance (histogram)
    compute_mahalanobis_distance         Average Mahalanobis distance

Weights (paper Section 6.2, derived via mutual information across 33 datasets):
    ALPHA = 0.5314  (separation / isolation index)
    BETA  = 0.4686  (Hellinger overlap)

Usage
-----
    from metrics import compute_asoi_from_predictions

    # y_pred: array of 1 (normal) and -1 (anomaly)
    score = compute_asoi_from_predictions(X, y_pred)
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm

# ---------- Paper weights (Section 6.2) ----------
ALPHA = 0.5314   # separation weight
BETA  = 0.4686   # Hellinger overlap weight


# =============================================================================
# UTILITIES
# =============================================================================

def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2-D (n_samples, d)."""
    arr = np.asarray(arr, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def normalise_array(array: np.ndarray) -> np.ndarray:
    """Min-max normalise a 1-D array to [0, 1]."""
    array = np.asarray(array, dtype=np.float64).reshape(-1, 1)
    if np.ptp(array) == 0:
        return np.zeros(len(array))
    return MinMaxScaler().fit_transform(array).flatten()


# =============================================================================
# COMPONENT 1 — ISOLATION INDEX  (Separation S, Eq. 8)
# =============================================================================

def compute_isolation_index(normal_data: np.ndarray,
                             anomaly_data: np.ndarray,
                             normalise: bool = True) -> float:
    """
    Isolation Index S — average distance of anomalous points to the
    centroid of normal data (Eq. 8 / Algorithm 1, steps 2-5).

    S captures how far anomalies deviate from the compact normal cluster.
    When ``normalise=True`` S is divided by the bounding-box diagonal,
    mapping the result to [0, 1].

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,  d)
    anomaly_data : np.ndarray (n_anomaly, d)
    normalise    : bool  Apply min-max normalisation (default True).

    Returns
    -------
    float  Isolation index in [0, 1] when normalised.
    """
    normal_data  = _to_2d(normal_data)
    anomaly_data = _to_2d(anomaly_data)

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return 0.0

    centroid_n = np.mean(normal_data, axis=0)
    distances  = np.linalg.norm(anomaly_data - centroid_n, axis=1)
    S          = float(np.mean(distances))

    if normalise:
        overall_min = np.minimum(normal_data.min(axis=0), anomaly_data.min(axis=0))
        overall_max = np.maximum(normal_data.max(axis=0), anomaly_data.max(axis=0))
        max_dist    = np.linalg.norm(overall_max - overall_min)
        S = S / max_dist if max_dist > 1e-8 else 0.0

    return S


# =============================================================================
# COMPONENT 2 — HELLINGER DISTANCE  (Overlap H, Eq. 9)
# =============================================================================

def compute_hellinger_distance(normal_data: np.ndarray,
                                anomaly_data: np.ndarray) -> float:
    """
    Average per-feature Hellinger distance H via Gaussian approximation
    (Eq. 9 / Algorithm 1, steps 6-8).

    H in [0, 1]:  0 = identical distributions,  1 = no overlap.
    In anomaly detection we want H close to 1 (well-separated).

    Per-feature PDFs are approximated by fitting a Gaussian (mean, std)
    and the Bhattacharyya coefficient is integrated numerically.

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,  d)
    anomaly_data : np.ndarray (n_anomaly, d)

    Returns
    -------
    float  Mean Hellinger distance across all d features.
    """
    normal_data  = _to_2d(normal_data)
    anomaly_data = _to_2d(anomaly_data)

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return 0.0

    distances = []
    for i in range(normal_data.shape[1]):
        mean_n = np.mean(normal_data[:, i])
        std_n  = max(np.std(normal_data[:, i]),  1e-8)
        mean_a = np.mean(anomaly_data[:, i])
        std_a  = max(np.std(anomaly_data[:, i]), 1e-8)

        lo  = min(normal_data[:, i].min(), anomaly_data[:, i].min())
        hi  = max(normal_data[:, i].max(), anomaly_data[:, i].max())
        buf = (hi - lo) * 0.2 if (hi - lo) > 1e-8 else 1.0
        x   = np.linspace(lo - buf, hi + buf, 1000)

        px = np.maximum(norm.pdf(x, mean_n, std_n), 0)
        qx = np.maximum(norm.pdf(x, mean_a, std_a), 0)

        bc = float(np.clip(np.trapz(np.sqrt(px * qx), x), 0.0, 1.0))
        distances.append(np.sqrt(1.0 - bc))

    return float(np.mean(distances)) if distances else 0.0


def compute_hellinger_distance_histogram(normal_data: np.ndarray,
                                          anomaly_data: np.ndarray,
                                          bins: int = None) -> float:
    """
    Hellinger distance via histogram bins (discrete approximation, Eq. 12).

    When ``bins=None`` the Rice Rule (Eq. 13) is applied:
        omega = ceil(2 * n^(1/3))

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,)
    anomaly_data : np.ndarray (n_anomaly,)
    bins         : int | None  Number of bins (None uses Rice Rule).

    Returns
    -------
    float  Hellinger distance in [0, 1].
    """
    normal_data  = np.asarray(normal_data,  dtype=np.float64).ravel()
    anomaly_data = np.asarray(anomaly_data, dtype=np.float64).ravel()

    if bins is None:
        n    = len(normal_data) + len(anomaly_data)
        bins = int(np.ceil(2 * n ** (1 / 3)))   # Rice Rule  Eq. 13

    p_hist, _ = np.histogram(normal_data,  bins=bins, density=True)
    q_hist, _ = np.histogram(anomaly_data, bins=bins, density=True)

    p_hist = (p_hist + 1e-10) / (p_hist + 1e-10).sum()
    q_hist = (q_hist + 1e-10) / (q_hist + 1e-10).sum()

    # Hellinger = norm(sqrt(p) - sqrt(q)) / sqrt(2)
    return float(np.linalg.norm(np.sqrt(p_hist) - np.sqrt(q_hist)) / np.sqrt(2))


# =============================================================================
# MAIN METRICS — ASI & ASOI
# =============================================================================

def compute_asi(normal_data: np.ndarray,
                anomaly_data: np.ndarray) -> float:
    """
    Anomaly Separation Index (ASI) — Eq. 3 of the paper.

    ASI = norm(mu_N - mu_A) / sqrt(sum_j sigma^2_pooled,j)

    Cohen's-d-style effect size adapted to high-dimensional data.
    Higher ASI -> greater separation between normal and anomalous distributions.

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,  d)
    anomaly_data : np.ndarray (n_anomaly, d)

    Returns
    -------
    float  ASI >= 0.
    """
    normal_data  = _to_2d(normal_data)
    anomaly_data = _to_2d(anomaly_data)

    if normal_data.shape[0] < 2 or anomaly_data.shape[0] < 2:
        return 0.0

    n_n, n_a = normal_data.shape[0], anomaly_data.shape[0]
    mu_n, mu_a = normal_data.mean(axis=0), anomaly_data.mean(axis=0)

    var_n = np.var(normal_data,  axis=0, ddof=1)   # sigma^2_{N,j}  Eq. 6
    var_a = np.var(anomaly_data, axis=0, ddof=1)   # sigma^2_{A,j}  Eq. 7

    # Pooled variance per feature  Eq. 5
    pooled_var  = ((n_n - 1) * var_n + (n_a - 1) * var_a) / (n_n + n_a - 2)
    pooled_var  = np.maximum(pooled_var, 1e-16)

    numerator   = np.linalg.norm(mu_n - mu_a)
    denominator = np.sqrt(np.sum(pooled_var))

    return float(numerator / denominator) if denominator > 1e-16 else 0.0


def compute_asoi(normal_data: np.ndarray,
                 anomaly_data: np.ndarray,
                 alpha: float = ALPHA,
                 beta: float  = BETA) -> float:
    """
    Anomaly Separation and Overlap Index (ASOI) — Eq. 14 / Algorithm 1.

    ASOI = alpha * S_norm + beta * H

    S_norm : normalised isolation index (separation component).
    H      : mean Hellinger distance    (overlap component).

    Higher ASOI -> better detection (well-separated, low-overlap predictions).

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,  d)
    anomaly_data : np.ndarray (n_anomaly, d)
    alpha        : float  Separation weight  (default ALPHA = 0.5314).
    beta         : float  Overlap weight     (default BETA  = 0.4686).

    Returns
    -------
    float  ASOI in [0, 1].
    """
    if len(normal_data) == 0 or len(anomaly_data) == 0:
        return 0.0

    S     = compute_isolation_index(normal_data, anomaly_data, normalise=True)
    H     = compute_hellinger_distance(normal_data, anomaly_data)
    score = alpha * S + beta * H

    return float(score) if np.isfinite(score) else 0.0


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def compute_asoi_from_predictions(X: np.ndarray,
                                   y_pred: np.ndarray,
                                   alpha: float = ALPHA,
                                   beta: float  = BETA) -> float:
    """
    Compute ASOI directly from a prediction array.

    Parameters
    ----------
    X      : np.ndarray (n_samples, d)   Feature matrix.
    y_pred : np.ndarray (n_samples,)     1 = normal, -1 = anomaly.
    alpha  : float  Separation weight  (default ALPHA = 0.5314).
    beta   : float  Overlap weight     (default BETA  = 0.4686).

    Returns
    -------
    float  ASOI in [0, 1], or -inf if all predictions are one class.
    """
    X, y_pred = np.asarray(X), np.asarray(y_pred)
    normal_data  = X[y_pred ==  1]
    anomaly_data = X[y_pred == -1]

    if normal_data.shape[0] == 0 or anomaly_data.shape[0] == 0:
        return -np.inf   # degenerate: all predicted as one class

    return compute_asoi(normal_data, anomaly_data, alpha=alpha, beta=beta)


def compute_asoi_for_model(model,
                            X,
                            alpha: float = ALPHA,
                            beta: float  = BETA) -> float:
    """
    Compute ASOI for a fitted scikit-learn-compatible anomaly detector.

    Calls ``model.predict(X)`` (must return 1 / -1) then evaluates ASOI
    on the resulting split.

    Parameters
    ----------
    model  : fitted estimator with a .predict() method.
    X      : array-like (n_samples, d).
    alpha  : float  Separation weight (default ALPHA = 0.5314).
    beta   : float  Overlap weight    (default BETA  = 0.4686).

    Returns
    -------
    float  ASOI score, or -inf on error.
    """
    try:
        X_np   = X.values if hasattr(X, 'values') else np.asarray(X)
        y_pred = model.predict(X_np)
        return compute_asoi_from_predictions(X_np, y_pred, alpha=alpha, beta=beta)
    except Exception as exc:
        print(f"[ASOI] Error computing score: {exc}")
        return -np.inf


def asoi_scorer(estimator, X, y=None) -> float:
    """
    Drop-in scorer for GridSearchCV / RandomizedSearchCV.

    ``y`` is accepted but intentionally ignored — ASOI is label-free.

    Example
    -------
    >>> GridSearchCV(estimator, param_grid, scoring=asoi_scorer)

    Returns
    -------
    float  ASOI score (higher = better), or -inf on failure.
    """
    return compute_asoi_for_model(estimator, X)


# =============================================================================
# SUPPLEMENTARY / COMPARATIVE METRICS
# =============================================================================

def compute_bhattacharyya_distance(normal_data: np.ndarray,
                                    anomaly_data: np.ndarray,
                                    bins: int = 10) -> float:
    """
    Bhattacharyya distance between normal and anomaly distributions
    (histogram approximation).

    B = -ln( sum sqrt(p_l * q_l) )

    Larger B -> distributions are more dissimilar.

    Parameters
    ----------
    normal_data  : np.ndarray (n_normal,)
    anomaly_data : np.ndarray (n_anomaly,)
    bins         : int  Number of histogram bins.

    Returns
    -------
    float  Bhattacharyya distance >= 0.
    """
    normal_data  = np.asarray(normal_data,  dtype=np.float64).ravel()
    anomaly_data = np.asarray(anomaly_data, dtype=np.float64).ravel()

    p_hist, _ = np.histogram(normal_data,  bins=bins, density=True)
    q_hist, _ = np.histogram(anomaly_data, bins=bins, density=True)

    p_hist = (p_hist + 1e-10) / (p_hist + 1e-10).sum()
    q_hist = (q_hist + 1e-10) / (q_hist + 1e-10).sum()

    bc = np.sum(np.sqrt(p_hist * q_hist))
    return float(-np.log(bc + 1e-10))


def compute_mahalanobis_distance(normal_data: np.ndarray,
                                  anomaly_data: np.ndarray,
                                  regularization: float = 1e-10) -> float:
    """
    Average Mahalanobis distance of anomaly points from the normal distribution.

    Uses the mean and covariance of normal data as the reference.
    A regularization term is added to the covariance diagonal to ensure
    invertibility, particularly in high-dimensional settings.

    Parameters
    ----------
    normal_data    : np.ndarray (n_normal,  d)
    anomaly_data   : np.ndarray (n_anomaly, d)
    regularization : float  Ridge term added to covariance diagonal.

    Returns
    -------
    float  Mean Mahalanobis distance of anomaly points.
    """
    normal_data  = _to_2d(normal_data)
    anomaly_data = _to_2d(anomaly_data)

    mean_n  = np.mean(normal_data, axis=0)
    cov_n   = np.cov(normal_data, rowvar=False)
    cov_n  += np.eye(cov_n.shape[0]) * regularization
    inv_cov = np.linalg.inv(cov_n)

    diff      = anomaly_data - mean_n
    distances = np.sqrt(np.einsum('ij,ij->i', np.dot(diff, inv_cov), diff))

    return float(np.mean(distances))


# =============================================================================
# QUICK SANITY-CHECK  (python metrics.py)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    normal  = rng.normal(loc=0.0, scale=1.0, size=(300, 5))   # compact normal
    far     = rng.normal(loc=5.0, scale=0.5, size=(30,  5))   # well-separated -> high ASOI
    overlap = rng.normal(loc=0.2, scale=1.0, size=(30,  5))   # overlapping    -> low  ASOI

    print("=" * 55)
    print("ASOI / ASI sanity check")
    print("=" * 55)
    print(f"  Well-separated  ASOI : {compute_asoi(normal, far):.4f}   <- expect high")
    print(f"  Overlapping     ASOI : {compute_asoi(normal, overlap):.4f}   <- expect low")
    print()
    print(f"  Well-separated  ASI  : {compute_asi(normal, far):.4f}")
    print(f"  Overlapping     ASI  : {compute_asi(normal, overlap):.4f}")
    print()
    print(f"  Isolation Index (sep): {compute_isolation_index(normal, far):.4f}")
    print(f"  Hellinger (sep)      : {compute_hellinger_distance(normal, far):.4f}")
    print(f"  Hellinger (overlap)  : {compute_hellinger_distance(normal, overlap):.4f}")
    print()
    print(f"  Bhattacharyya (sep)  : {compute_bhattacharyya_distance(normal[:,0], far[:,0]):.4f}")
    print(f"  Mahalanobis   (sep)  : {compute_mahalanobis_distance(normal, far):.4f}")
    print()

    # Test from-predictions helper
    X_all  = np.vstack([normal, far])
    y_pred = np.array([1] * 300 + [-1] * 30)
    print(f"  compute_asoi_from_predictions : {compute_asoi_from_predictions(X_all, y_pred):.4f}")
