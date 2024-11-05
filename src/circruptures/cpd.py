from typing import Tuple

import numpy as np
from numba import njit

from .stats import circular_mean_1d


@njit
def init_centroids(signal: np.ndarray, n_states: int) -> np.ndarray:
    """Initialize centroids.

    Args:
        signal (np.ndarray): A 2D array of shape (n_samples, n_dims) representing the input data.
        n_states (int): The number of centroids (clusters) to initialize.

    Returns:
        np.ndarray: A 2D array of shape (n_centroids, n_dims) representing the initialized centroids.
    """
    n_dims = signal.shape[1]
    centroids = np.empty(shape=(n_states, n_dims), dtype=np.float64)
    chunks = np.array_split(signal, n_states)

    for k_state in range(n_states):
        for k_dim in range(n_dims):
            centroids[k_state, k_dim] = circular_mean_1d(chunks[k_state][:, k_dim])
    return centroids


@njit
def get_best_previous_state_and_soc(
    soc_vec: np.ndarray, end_state: int, penalty: float
) -> Tuple[int, float]:
    """Update the sum of costs (soc) and the best previous state given the end state and a penalty.

    Parameters:
    - soc_vec (np.ndarray): Vector of sums of costs (soc), shape (n_states,).
    - end_state (int): The end state for which the update is performed.
    - penalty (float): Penalty value for the update.

    Returns:
    - Tuple[int, float]: Tuple containing the best previous state and the updated sum of costs.
    """
    n_states = soc_vec.shape[0]
    best_previous_state = end_state
    best_soc = soc_vec[best_previous_state]
    for k_state in range(n_states):
        if k_state != end_state:
            soc = soc_vec[k_state]
            if soc + penalty < best_soc:
                best_previous_state = k_state
                best_soc = soc + penalty
    return best_previous_state, best_soc


@njit
def get_state_sequence(costs: np.ndarray, penalty: float) -> np.ndarray:
    """Return the optimal state sequence for a given cost array and penalty.

    Parameters:
    - costs (np.ndarray): Array of cost values, shape (n_samples, n_states).
    - penalty (float): Penalty value.

    Returns:
    - np.ndarray: Optimal state sequence, shape (n_samples,).
    """
    n_samples, n_states = costs.shape
    soc_array = np.empty((n_samples + 1, n_states), dtype=np.float64)
    state_array = np.empty((n_samples + 1, n_states), dtype=np.int32)
    soc_array[0] = 0
    state_array[0] = -1

    # Forward loop
    for end in range(1, n_samples + 1):
        for k_state in range(n_states):
            best_state, best_soc = get_best_previous_state_and_soc(
                soc_vec=soc_array[end - 1], end_state=k_state, penalty=penalty
            )
            soc_array[end, k_state] = best_soc + costs[end - 1, k_state]
            state_array[end, k_state] = best_state

    # Backtracking
    end = n_samples
    state = np.argmin(soc_array[end])
    states = np.empty(n_samples, dtype=np.int32)
    while (state > -1) and (end > 0):
        states[end - 1] = state
        state = state_array[end, state]
        end -= 1
    return states


@njit
def dist_func(x: np.ndarray, y: np.ndarray) -> float:
    # x, shape (2,)
    # y, shape (2,)
    diff = np.abs(x - y)
    return np.sum(np.fmin(diff, 2 * np.pi - diff))


@njit
def compute_all_costs(signal, means):
    n_samples = signal.shape[0]
    n_states = means.shape[0]
    costs = np.empty((n_samples, n_states), dtype=np.float64)
    for k_state in range(n_states):
        for k_sample in range(n_samples):
            costs[k_sample, k_state] = dist_func(signal[k_sample], means[k_state]) ** 2
    return costs


def get_bkps(signal, penalty=1, n_states=10, return_approx=False):
    centroids = init_centroids(signal, n_states)
    costs = compute_all_costs(signal, centroids)
    states = get_state_sequence(costs, penalty)
    bkps = np.nonzero(np.diff(states))[0]
    n_samples = signal.shape[0]
    bkps = np.append(bkps, n_samples)
    if return_approx:
        approx = centroids[states]
        return bkps, approx
    return bkps
