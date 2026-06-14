# SDE PyTorch Project

This mini–project demonstrates how to simulate and model stochastic differential equations (SDEs) and how to use a neural network to learn the underlying drift function. It contains:

* `sde_simulation.py` – functions to simulate one‐dimensional SDEs such as the Ornstein–Uhlenbeck (OU) process using the Euler–Maruyama method.
* `train_nn.py` – a training script that uses PyTorch to approximate the drift function of an OU process from simulated data. It trains a small neural network on the derivative data and produces a plot comparing the learned drift to the true drift.
* `requirements.txt` – a list of Python packages used in this project.

## How to run

1. Create a virtual environment (optional but recommended) and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. To generate example simulation data and train the neural network, run:

   ```bash
   python train_nn.py
   ```

   The script will:
   - Simulate multiple Ornstein–Uhlenbeck trajectories.
   - Compute approximate drift values from the simulated data.
   - Train a neural network to learn the drift function.
   - Save a figure in `figures/drift_fit.png` showing the true vs. learned drift.
   - Save a figure in `figures/trajectories.png` with a few sample SDE trajectories.

3. Examine the figures in the `figures` directory and review the output printed to the console for training progress and loss values.

## Project structure

```
sde_pytorch_project/
├── README.md        # this file
├── requirements.txt # Python dependencies
├── sde_simulation.py
├── train_nn.py
└── figures/
    ├── drift_fit.png
    └── trajectories.png
```

## Notes

This project is intentionally small and focused. The neural network learns the drift of an OU process from simulated data. You can extend these functions to other types of SDEs (e.g. geometric Brownian motion) or experiment with different network architectures. 
