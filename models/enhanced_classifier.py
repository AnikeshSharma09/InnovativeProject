"""
Enhanced Biomimetic Classifier Wrapper and Geometric Invariant Feature Extraction.

Extracts 20 physiological and geometric invariants from hand landmark coordinates:
- 7 Inter-fingertip Euclidean distances
- 5 Tip-to-wrist Euclidean distances
- 5 Finger curl and extension ratios
- 3 Inter-finger bone spread angles (vector cosines)

Seamlessly integrates with scikit-learn ensemble models and predictor pipelines.
"""
from typing import Tuple, List, Optional
import numpy as np


def compute_geometric_invariants(X_63: np.ndarray) -> np.ndarray:
    """
    Computes 20 physical geometric invariant features:
    - 7 Inter-fingertip distances
    - 5 Tip-to-wrist distances
    - 5 Finger extension / curl ratios
    - 3 Finger spread angles (cosines between rays)

    Handles both (N, 63) batches and single (63,) vectors.
    Returns (N, 83) feature matrix.
    """
    is_1d = False
    if X_63.ndim == 1:
        is_1d = True
        X_63 = X_63.reshape(1, -1)

    N = X_63.shape[0]
    pts = X_63.reshape(N, 21, 3)

    # 1. Inter-fingertip Euclidean distances
    d_ti = np.linalg.norm(pts[:, 4] - pts[:, 8], axis=-1, keepdims=True)
    d_tm = np.linalg.norm(pts[:, 4] - pts[:, 12], axis=-1, keepdims=True)
    d_tr = np.linalg.norm(pts[:, 4] - pts[:, 16], axis=-1, keepdims=True)
    d_tp = np.linalg.norm(pts[:, 4] - pts[:, 20], axis=-1, keepdims=True)
    d_im = np.linalg.norm(pts[:, 8] - pts[:, 12], axis=-1, keepdims=True)
    d_mr = np.linalg.norm(pts[:, 12] - pts[:, 16], axis=-1, keepdims=True)
    d_rp = np.linalg.norm(pts[:, 16] - pts[:, 20], axis=-1, keepdims=True)

    # 2. Tip-to-wrist distances
    d_w_t = np.linalg.norm(pts[:, 4] - pts[:, 0], axis=-1, keepdims=True)
    d_w_i = np.linalg.norm(pts[:, 8] - pts[:, 0], axis=-1, keepdims=True)
    d_w_m = np.linalg.norm(pts[:, 12] - pts[:, 0], axis=-1, keepdims=True)
    d_w_r = np.linalg.norm(pts[:, 16] - pts[:, 0], axis=-1, keepdims=True)
    d_w_p = np.linalg.norm(pts[:, 20] - pts[:, 0], axis=-1, keepdims=True)

    # 3. Finger extension ratios (tip-to-wrist / PIP-to-wrist)
    ext_t = d_w_t / (np.linalg.norm(pts[:, 2] - pts[:, 0], axis=-1, keepdims=True) + 1e-6)
    ext_i = d_w_i / (np.linalg.norm(pts[:, 6] - pts[:, 0], axis=-1, keepdims=True) + 1e-6)
    ext_m = d_w_m / (np.linalg.norm(pts[:, 10] - pts[:, 0], axis=-1, keepdims=True) + 1e-6)
    ext_r = d_w_r / (np.linalg.norm(pts[:, 14] - pts[:, 0], axis=-1, keepdims=True) + 1e-6)
    ext_p = d_w_p / (np.linalg.norm(pts[:, 18] - pts[:, 0], axis=-1, keepdims=True) + 1e-6)

    # 4. Finger spread angles (cosines between finger bone rays)
    v_i = pts[:, 8] - pts[:, 5]
    v_m = pts[:, 12] - pts[:, 9]
    v_r = pts[:, 16] - pts[:, 13]
    v_p = pts[:, 20] - pts[:, 17]

    norm_i = np.linalg.norm(v_i, axis=-1, keepdims=True) + 1e-6
    norm_m = np.linalg.norm(v_m, axis=-1, keepdims=True) + 1e-6
    norm_r = np.linalg.norm(v_r, axis=-1, keepdims=True) + 1e-6
    norm_p = np.linalg.norm(v_p, axis=-1, keepdims=True) + 1e-6

    cos_im = np.sum(v_i * v_m, axis=-1, keepdims=True) / (norm_i * norm_m)
    cos_mr = np.sum(v_m * v_r, axis=-1, keepdims=True) / (norm_m * norm_r)
    cos_rp = np.sum(v_r * v_p, axis=-1, keepdims=True) / (norm_r * norm_p)

    out = np.hstack([
        X_63,
        d_ti, d_tm, d_tr, d_tp, d_im, d_mr, d_rp,
        d_w_t, d_w_i, d_w_m, d_w_r, d_w_p,
        ext_t, ext_i, ext_m, ext_r, ext_p,
        cos_im, cos_mr, cos_rp
    ])

    return out


class EnhancedSignClassifier:
    """
    Wrapper model that automatically transforms incoming 63 raw landmarks
    into the full 83 physical geometric invariant features before inference.
    """
    def __init__(self, base_estimator, classes):
        self.base_estimator = base_estimator
        self.classes_ = np.array(classes)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == 63:
            X_aug = compute_geometric_invariants(X)
        else:
            X_aug = X
        return self.base_estimator.predict_proba(X_aug)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == 63:
            X_aug = compute_geometric_invariants(X)
        else:
            X_aug = X
        return self.base_estimator.predict(X_aug)
