from typing import Tuple, Union

import numpy as np
from numba import njit

from .stats import circular_mean_1d


@njit
def init_centroids(signal: np.ndarray, n_states: int) -> np.ndarray:
    """Initialize centroids.

    Args:
        signal (np.ndarray): A 2D array of shape (n_samples, n_dims) representing the
        input data.
        n_states (int): The number of centroids (clusters) to initialize.

    Returns:
        np.ndarray: A 2D array of shape (n_centroids, n_dims) representing the
        initialized centroids.
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
    """Update the sum of costs (soc) and the best previous state given the end state and
    a penalty.

    Parameters:
    - soc_vec (np.ndarray): Vector of sums of costs (soc), shape (n_states,).
    - end_state (int): The end state for which the update is performed.
    - penalty (float): Penalty value for the update.

    Returns:
    - Tuple[int, float]: Tuple containing the best previous state and the updated sum of
    costs.
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
        best_state_if_change = np.argmin(soc_array[end - 1])
        current_min_soc_with_pen = soc_array[end - 1, best_state_if_change] + penalty
        for k_state in range(n_states):
            cost_value = costs[end - 1, k_state]
            if soc_array[end - 1, k_state] < current_min_soc_with_pen:  # no change
                soc_array[end, k_state] = soc_array[end - 1, k_state] + cost_value
                state_array[end, k_state] = k_state
            else:  # with change
                soc_array[end, k_state] = current_min_soc_with_pen + cost_value
                state_array[end, k_state] = best_state_if_change

            # best_state, best_soc = get_best_previous_state_and_soc(
            #     soc_vec=soc_array[end - 1], end_state=k_state, penalty=penalty
            # )
            # soc_array[end, k_state] = best_soc + costs[end - 1, k_state]
            # state_array[end, k_state] = best_state

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
    """Computes the circular (angular) distance between two points on a circle.

    Given two points `x` and `y` (each as 1D NumPy arrays of shape (n_dims,)), this
    function calculates the sum of the minimal angular distances between corresponding
    elements, considering the periodicity of the circle (with period 2π).
    Parameters
    ----------
    x : np.ndarray
        A 1D NumPy array of shape (n_dims,) representing the first point in angular
        coordinates (radians).
    y : np.ndarray
        A 1D NumPy array of shape (n_dims,) representing the second point in angular
        coordinates (radians).
    Returns
    -------
    float
        The sum of the minimal angular distances between the corresponding elements of
        `x` and `y`.
    """

    # x, shape (n_dims,)
    # y, shape (n_dims,)
    diff = np.abs(x - y)
    return np.sum(np.fmin(diff, 2 * np.pi - diff))


@njit
def compute_all_costs(signal, means):
    """
    Compute the squared distance costs between each sample in the signal and each state
    mean.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, n_dims)
        The input data, where each row corresponds to a sample and each column to a
        feature dimension.
    means : np.ndarray, shape (n_states, n_dims)
        The mean vectors for each state, where each row corresponds to a state mean.

    Returns
    -------
    costs : np.ndarray, shape (n_samples, n_states)
        The matrix of squared distances, where costs[i, j] is the squared distance
        between signal[i] and means[j] as computed by `dist_func`.

    Notes
    -----
    Requires a distance function `dist_func` to be defined in the scope, which computes
    the distance between two vectors.
    """
    # signal, shape (n_samples, n_dims)
    # means, shape (n_states, n_dims)
    # costs, shape (n_samples, n_states)
    n_samples = signal.shape[0]
    n_states = means.shape[0]
    costs = np.empty((n_samples, n_states), dtype=np.float64)
    for k_state in range(n_states):
        for k_sample in range(n_samples):
            costs[k_sample, k_state] = dist_func(signal[k_sample], means[k_state]) ** 2
    return costs


def get_bkps(
    signal: np.ndarray,
    penalty: float = 1,
    n_states: int = 10,
    return_approx: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute change points in a circular signal

    Args:
        signal (np.ndarray): Input data of shape (n_samples, n_dims).
        penalty (float, optional): Penalty value for state transitions. Default is 1.
        n_states (int, optional): Number of states (clusters) to use. Default is 10.
        return_approx (bool, optional): If True, also return the approximated signal. Default is False.

    Returns:
        np.ndarray: Indices of breakpoints.
        If return_approx is True, also returns the approximated signal (np.ndarray).
    """
    centroids = init_centroids(signal, n_states)  # shape (n_states, n_dims)
    costs = compute_all_costs(signal, centroids)  # shape (n_samples, n_states)
    states = get_state_sequence(costs, penalty)  # shape (n_samples,)
    bkps = np.nonzero(np.diff(states))[0]  # indices of change points
    n_samples = signal.shape[0]
    bkps = np.append(bkps, n_samples)
    if return_approx:
        approx = centroids[states]
        return bkps, approx
    return bkps
