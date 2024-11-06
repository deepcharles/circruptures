from itertools import tee

import numpy as np
from numba import njit
from scipy.stats import circvar, wasserstein_distance
from .stats import circular_mean_axis0

# def triwise(iterable):
#     a, b = tee(iterable)
#     a, c = tee(iterable)
#     next(b, None)
#     next(c, None)
#     next(c, None)
#     return zip(a, b, c)


# def convert_bkps_to_distrib(signal: np.ndarray, bkps: np.ndarray, jump:int=1):
#     n_samples = bkps[-1]
#     assert signal.shape[0] == n_samples, "Signal length and change points do not match."

#     stats = list()
#     bkps = np.array(bkps)
#     bkps_with_0 = [0] + bkps.tolist()
#     for left, mid, right in triwise(bkps_with_0):
#         left_var = circvar(signal[left:mid:jump], axis=0)
#         right_var = circvar(signal[mid:right:jump], axis=0)
#         all_var = circvar(signal[left:right:jump], axis=0)
#         stats.append(
#             (
#                 all_var * (right - left)
#                 - left_var * (mid - left)
#                 - right_var * (right - mid)
#             ).sum()
#         )
#     stats = np.array(stats)
#     stats /= stats.sum()
#     distrib = np.zeros(n_samples)
#     distrib[bkps[:-1]] = stats
#     return distrib


@njit
def select_by_peak_distance(peaks, priority, distance):
    peaks_size = peaks.shape[0]
    # Round up because actual peak distance can only be natural number
    keep = np.ones(peaks_size, dtype=np.uint8)  # Prepare array of flags
    # Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
    # for `peaks` sorted by position). This allows to iterate `peaks` and `keep`
    # with `j` by order of `priority` while still maintaining the ability to
    # step to neighbouring peaks with (`j` + 1) or (`j` - 1).
    priority_to_position = np.argsort(priority)

    # Highest priority first -> iterate in reverse order (decreasing)
    for i in range(peaks_size - 1, -1, -1):
        # "Translate" `i` to `j` which points to current peak whose
        # neighbours are to be evaluated
        j = priority_to_position[i]
        if keep[j] == 0:
            # Skip evaluation for peak already marked as "don't keep"
            continue

        k = j - 1
        # Flag "earlier" peaks for removal until minimal distance is exceeded
        while 0 <= k and peaks[j] - peaks[k] < distance:
            keep[k] = 0
            k -= 1

        k = j + 1
        # Flag "later" peaks for removal until minimal distance is exceeded
        while k < peaks_size and peaks[k] - peaks[j] < distance:
            keep[k] = 0
            k += 1
    return keep


@njit
def convert_bkps_to_distrib(signal: np.ndarray, bkps: np.ndarray, width: int):
    distrib = np.zeros_like(signal[:, 0])
    n_samples = signal.shape[0]
    for bkp in bkps[:-1]:
        left_mean = circular_mean_axis0(signal[max(0, bkp - width) : bkp])
        right_mean = circular_mean_axis0(signal[bkp : min(bkp + width, n_samples)])
        diff_vec = right_mean - left_mean
        distrib[bkp] = np.sum(np.square(diff_vec))
    keep_mask = select_by_peak_distance(bkps[:-1], distrib[bkps[:-1]], width)
    distrib[bkps[:-1]] *= keep_mask
    distrib /= distrib.sum()
    return distrib


def distance_between_bkps(
    signal1: np.ndarray,
    bkps1: np.ndarray,
    signal2: np.ndarray,
    bkps2: np.ndarray,
    width: int
) -> float:
    assert bkps1[-1] == bkps2[-1], "Both signals should have the same length."
    distrib1 = convert_bkps_to_distrib(signal=signal1, bkps=bkps1, width=width)
    distrib2 = convert_bkps_to_distrib(signal=signal2, bkps=bkps2, width=width)
    n_samples = bkps1[-1]
    support = np.linspace(0, 1, n_samples)
    return wasserstein_distance(support, support, distrib1, distrib2)
