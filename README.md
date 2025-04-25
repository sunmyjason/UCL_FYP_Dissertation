# MEng Final Project – Reinforcement Learning for Market Making

This repository contains the code for the UCL MEng Computer Science final year project titled:

**"Reinforcement Learning for High-Frequency Market Making Strategies with Realistic Limit Order Book Simulations"**

Author: *21153809*
Supervisor: *Dr Silvia Bartolucci*  
Submission Date: *April 2025*

---

## Folder Overview

- `calibration/`  
  Contains scripts for calibrating mid-price, fill probability, and order arrival processes using empirical LOB data.

- `mbt_gym/`  
  Custom extension of the `mbt-gym` framework to implement a realistic, RL-compatible limit order book simulation environment.

- `rl_notebooks/`  
  Jupyter notebooks used for developing, training, and evaluating RL agents (PPO, A2C, SAC) and for running Optuna-based hyperparameter tuning.

---

## Note on Data

Raw market data used for calibration and evaluation (~30GB) is **not included** in this archive due to size and licensing constraints (LOBSTER).
