"""Train a neural network to approximate the drift function of an SDE.

This script simulates an Ornstein–Uhlenbeck process using the Euler–Maruyama method, computes
finite‐difference estimates of the drift (i.e. (X_{n+1}-X_n)/dt), and trains a small feedforward
neural network to learn the drift function from data. After training, it produces plots
comparing the learned drift to the true drift and visualizes a few sample trajectories.

Run this script from the project root:

```
python train_nn.py
```

The resulting figures will be saved in the `figures/` directory.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sde_simulation import simulate_ou


def main() -> None:
    """Simulate an OU process and train a neural network to learn its drift."""
    # Parameters for the Ornstein–Uhlenbeck process
    theta = 2.0  # mean-reversion rate
    mu = 0.0     # long-term mean
    sigma = 0.1  # volatility (smaller volatility yields less noisy derivatives)
    x0 = 0.5     # initial condition
    dt = 0.01    # time step size
    # Use fewer steps/paths for easier training on a reasonable dataset size
    n_steps = 500   # number of time steps
    n_paths = 20    # number of independent sample paths

    # Simulate OU trajectories
    print("Simulating Ornstein–Uhlenbeck process...")
    X = simulate_ou(theta=theta, mu=mu, sigma=sigma, x0=x0, dt=dt,
                    n_steps=n_steps, n_paths=n_paths, random_seed=42)

    # Compute finite‐difference approximations of drift: (X_{n+1} - X_n) / dt
    dX = (X[:, 1:] - X[:, :-1]) / dt
    X_mid = X[:, :-1]

    # Flatten the data; each row is a pair (x_n, dx/dt)
    X_flat = X_mid.reshape(-1, 1)
    dX_dt_flat = dX.reshape(-1, 1)

    # Convert to torch tensors
    X_torch = torch.from_numpy(X_flat.astype(np.float32))
    y_torch = torch.from_numpy(dX_dt_flat.astype(np.float32))

    # Define a simple feedforward neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Train the network
    epochs = 500
    print("Training neural network to learn the drift function...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_torch)
        loss = loss_fn(y_pred, y_torch)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss {loss.item():.6f}")

    # Evaluate the learned drift function on a grid
    xs = np.linspace(X_flat.min(), X_flat.max(), 300, dtype=np.float32).reshape(-1, 1)
    xs_tensor = torch.from_numpy(xs)
    model.eval()
    with torch.no_grad():
        drift_pred = model(xs_tensor).numpy().flatten()

    # True drift function for OU: θ(μ − x)
    drift_true = theta * (mu - xs.flatten())

    # Create figures directory if necessary
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Plot true vs learned drift
    plt.figure()
    plt.plot(xs, drift_true, label='True drift', color='blue')
    plt.plot(xs, drift_pred, label='Learned drift', color='red', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('drift')
    plt.legend()
    plt.title('Drift function: True vs Learned')
    drift_plot_path = os.path.join(fig_dir, 'drift_fit.png')
    plt.savefig(drift_plot_path, dpi=200)
    plt.close()

    # Plot sample trajectories
    plt.figure()
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    for i in range(min(10, n_paths)):
        plt.plot(t, X[i], alpha=0.8)
    plt.xlabel('Time t')
    plt.ylabel('X_t')
    plt.title('Sample Ornstein–Uhlenbeck trajectories')
    traj_plot_path = os.path.join(fig_dir, 'trajectories.png')
    plt.savefig(traj_plot_path, dpi=200)
    plt.close()

    print(f"Saved drift plot to {drift_plot_path}")
    print(f"Saved trajectories plot to {traj_plot_path}")


if __name__ == '__main__':
    main()