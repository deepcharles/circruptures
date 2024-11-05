# Change point detection for circular time series

## Install

```bash
python -m pip install .
```


## Example

```Python
import matplotlib.pyplot as plt
import numpy as np

from circruptures import (
    get_bkps,
    estimate_var,
    distance_between_bkps,
    convert_bkps_to_distrib,
    get_approx,
)


# Assume data are in a variable named `signal`

# compute change points
penalty_factor = 14_000
n_samples = signal.shape[0]
variance = estimate_var(signal, n_segments=50)
bkps = get_bkps(
    signal,
    penalty=penalty_factor * np.log(n_samples) * variance,
    return_approx=False,
    n_states=20,
)

# Plot signal and its approximation
approx = get_approx(signal, bkps)
for dim in range(2):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal[:, dim])
    ax.plot(approx[:, dim])
    ax.set_xmargin(0)
    _ = ax.set_ylim(-np.pi, np.pi)
```
