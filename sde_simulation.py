"""Simulate one-dimensional stochastic differential equations (SDEs).

This module provides functions to simulate 1D stochastic differential equations using the
Euler–Maruyama method. It includes a convenience function for the Ornstein–Uhlenbeck process
and a generic simulator that accepts user-provided drift and diffusion functions.

Examples
--------
Simulate an Ornstein–Uhlenbeck process with parameters `theta`, `mu`, and `sigma`:

>>> from sde_simulation import simulate_ou
>>> import matplotlib.pyplot as plt
>>> X = simulate_ou(theta=1.0, mu=0.0, sigma=0.3, x0=0.5, dt=0.01, n_steps=1000, n_paths=5)
>>> for i in range(5):
...     plt.plot(X[i])
>>> plt.show()

Alternatively, simulate a custom SDE defined by drift and diffusion functions:

>>> def drift(x): return -x
>>> def diffusion(x): return 1.0
>>> X = simulate_sde(drift, diffusion, x0=0.0, dt=0.01, n_steps=1000, n_paths=10)
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, Optional


def simulate_ou(theta: float,
                mu: float,
                sigma: float,
                x0: float,
                dt: float,
                n_steps: int,
                n_paths: int = 1,
                random_seed: Optional[int] = None) -> np.ndarray:
    """Simulate the Ornstein–Uhlenbeck (OU) process via Euler–Maruyama.

    The OU process is given by

        dX_t = θ (μ − X_t) dt + σ dW_t,

    where θ is the mean-reversion rate, μ is the long-term mean, and σ is the volatility.

    Parameters
    ----------
    theta : float
        Mean-reversion rate (θ).
    mu : float
        Long-term mean (μ).
    sigma : float
        Volatility parameter (σ).
    x0 : float
        Initial value of the process at time zero.
    dt : float
        Time step size.
    n_steps : int
        Number of time steps to simulate (the number of discrete increments).
    n_paths : int, optional
        Number of independent sample paths to simulate. Default is 1.
    random_seed : int, optional
        Optional seed for NumPy's random number generator to ensure reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape `(n_paths, n_steps+1)` containing the simulated paths. Each row
        corresponds to a single sample path. The first column holds the initial values.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    # Initialise array to store paths
    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = x0

    # Precompute constants
    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        # Brownian increments
        dW = rng.standard_normal(n_paths) * sqrt_dt
        # Euler–Maruyama update
        X[:, i + 1] = X[:, i] + theta * (mu - X[:, i]) * dt + sigma * dW
    return X


def simulate_sde(drift: Callable[[np.ndarray], np.ndarray],
                 diffusion: Callable[[np.ndarray], np.ndarray],
                 x0: float,
                 dt: float,
                 n_steps: int,
                 n_paths: int = 1,
                 random_seed: Optional[int] = None) -> np.ndarray:
    """Simulate a one-dimensional Itô SDE via Euler–Maruyama.

    The SDE has the form

        dX_t = a(X_t) dt + b(X_t) dW_t,

    where `a` is the drift function and `b` is the diffusion function.

    Parameters
    ----------
    drift : Callable[[np.ndarray], np.ndarray]
        Function returning the drift term `a(x)` given the current state (element-wise).
    diffusion : Callable[[np.ndarray], np.ndarray]
        Function returning the diffusion term `b(x)` given the current state (element-wise).
    x0 : float
        Initial condition for all sample paths.
    dt : float
        Time step size.
    n_steps : int
        Number of time steps to simulate.
    n_paths : int, optional
        Number of independent sample paths. Default is 1.
    random_seed : int, optional
        Optional seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape `(n_paths, n_steps+1)` with simulated paths.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = x0
    sqrt_dt = np.sqrt(dt)
    for i in range(n_steps):
        x_curr = X[:, i]
        dW = rng.standard_normal(n_paths) * sqrt_dt
        X[:, i + 1] = x_curr + drift(x_curr) * dt + diffusion(x_curr) * dW
    return X