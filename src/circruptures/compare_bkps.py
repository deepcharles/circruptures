from itertools import tee

import numpy as np
from scipy.stats import circvar, wasserstein_distance


def triwise(iterable):
    a, b = tee(iterable)
    a, c = tee(iterable)
    next(b, None)
    next(c, None)
    next(c, None)
    return zip(a, b, c)


def convert_bkps_to_distrib(signal: np.ndarray, bkps: np.ndarray, jump:int=1):
    n_samples = bkps[-1]
    assert signal.shape[0] == n_samples, "Signal length and change points do not match."

    stats = list()
    bkps = np.array(bkps)
    bkps_with_0 = [0] + bkps.tolist()
    for left, mid, right in triwise(bkps_with_0):
        left_var = circvar(signal[left:mid:jump], axis=0)
        right_var = circvar(signal[mid:right:jump], axis=0)
        all_var = circvar(signal[left:right:jump], axis=0)
        stats.append(
            (
                all_var * (right - left)
                - left_var * (mid - left)
                - right_var * (right - mid)
            ).sum()
        )
    stats = np.array(stats)
    stats /= stats.sum()
    distrib = np.zeros(n_samples)
    distrib[bkps[:-1]] = stats
    return distrib

def distance_between_bkps(
    signal1: np.ndarray, bkps1: np.ndarray, signal2: np.ndarray, bkps2: np.ndarray,jump:int=1
) -> float:
    assert bkps1[-1]==bkps2[-1], "Both signals should have the same length."
    distrib1 = convert_bkps_to_distrib(signal=signal1, bkps=bkps1, jump=jump)
    distrib2 = convert_bkps_to_distrib(signal=signal2, bkps=bkps2, jump=jump)
    n_samples = bkps1[-1]
    support = np.linspace(0, 1, n_samples)
    return wasserstein_distance(support, support, distrib1, distrib2)
