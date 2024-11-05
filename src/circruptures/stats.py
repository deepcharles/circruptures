from typing import Tuple

import numpy as np
from numba import njit


@njit
def circular_mean_1d(angles_in_rad: np.ndarray) -> float:
    # angles_in_rad, shape (n_samples,)
    x = np.sum(np.cos(angles_in_rad))
    y = np.sum(np.sin(angles_in_rad))
    return np.arctan2(y, x)


@njit
def circular_var_1d(angles_in_rad: np.ndarray) -> float:
    # angles_in_rad, shape (n_samples,)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html
    x = np.mean(np.cos(angles_in_rad))
    y = np.mean(np.sin(angles_in_rad))
    return 1 - np.sqrt(x**2 + y**2)


@njit
def estimate_var_(signal: np.ndarray, n_segments: int = 100) -> np.ndarray:
    # signal, shape (n_samples, n_dims)
    # output, shape (n_segments, n_dims)
    _, n_dims = signal.shape
    variances = np.empty(shape=(n_segments, n_dims), dtype=np.float64)
    chunks = np.array_split(signal, n_segments)
    for k_seg in range(n_segments):
        for k_dim in range(n_dims):
            variances[k_seg, k_dim] = circular_var_1d(chunks[k_seg][:, k_dim])
    return variances


def estimate_var(signal, n_segments=100) -> float:
    # signal, shape (n_samples, n_dims)
    variances = estimate_var_(signal=signal, n_segments=n_segments)
    return np.median(variances, axis=0).sum(axis=0)
