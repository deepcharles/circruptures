import numpy as np
from scipy.stats import circmean
def get_approx(signal: np.ndarray, bkps: np.ndarray)->np.ndarray:
    # signal, shape (n_samples, n_dims)
    approx = np.empty_like(signal)
    n_bkps = len(bkps)
    start = 0
    for end in bkps:
        approx[start:end] = circmean(signal[start:end], axis=0, low=-np.pi, high=np.pi)
        start = end
    return approx