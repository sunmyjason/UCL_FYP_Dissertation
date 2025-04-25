# Consolidated import statements
import sys
sys.path.append("../")  # Add repo path

# Standard Libraries
import os
import shutil
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import optuna

# Gym and Stable Baselines3
import gym
from stable_baselines3 import A2C, PPO, TD3, SAC, DDPG
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# MBT Gym imports
from mbt_gym.gym.wrappers import ReduceStateSizeWrapper
from mbt_gym.gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.TradingEnvironment import TradingEnvironment
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.rewards.RewardFunctions import PnL, CjMmCriterion
from mbt_gym.stochastic_processes.midprice_models import *
from mbt_gym.stochastic_processes.arrival_models import *
from mbt_gym.stochastic_processes.fill_probability_models import *
from mbt_gym.gym.ModelDynamics import LimitOrderModelDynamics
from mbt_gym.gym.helpers.plotting import *
from mbt_gym.gym.helpers.visualize_return import *
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.gym.index_names import *
from mbt_gym.agents.SbAgent import SbAgent
from mbt_gym.agents.BaselineAgents import *

# Additional Libraries
from sklearn.metrics import mean_squared_error
import optuna.visualization as vis

def get_env_myModel(num_trajectories: int = 1):
    # Generic parameters
    initial_price = 100
    terminal_time = 1.0
    n_steps = 4000
    initial_inventory = 0
    max_inventory = 100

    # Calibrated midprice model parameters
    long_term_drift = 0.0
    sigma = 0.2 # GBM long-term volatility
    zeta = 0.00214  # mean reversion speed
    eta = 0.02207 # Short-term alpha volatility
    epsilon_plus = 0.000219
    epsilon_minus = -0.000184  # Jump size of sell MOs

    # Calibrated arrival model parameters (FROM PAPER)
    baseline_arrival_rate = 29.16 
    mean_reversion_speed = 140.05
    self_excitation_factor = 64.16
    cross_excitation_factor = 55.73

    # Calibrated fill probability model parameters
    fill_exponent = 1.7348 # Power law fill function exponent
    fill_multiplier = 0.1377 # Power law fill function multiplier

    # Reward function parameters
    alpha = 0.001 # terminal inventory penalty (fees of market orders and walking the book)
    phi = 0.5 # running inventory penalty parameter

    midprice_model = CustomAlphaMidpriceModel(long_term_drift = long_term_drift, 
                                              volatility = sigma,
                                              zeta = zeta, # ζ
                                              eta = eta, # η
                                              epsilon_plus = epsilon_plus, # ε^+
                                              epsilon_minus = epsilon_minus, # ε^-
                                              initial_price = initial_price,
                                              terminal_time = terminal_time,
                                              step_size = 1/n_steps,
                                              num_trajectories = num_trajectories)
    arrival_model = CustomArrivalModel(baseline_arrival_rate=baseline_arrival_rate,
                                        mean_reversion_speed=mean_reversion_speed,
                                        self_excitation_factor=self_excitation_factor,
                                        cross_excitation_factor=cross_excitation_factor,
                                        step_size=1/n_steps,
                                        terminal_time=terminal_time,
                                        num_trajectories=num_trajectories)
    fill_probability_model = CustomPowerFillFunction(fill_exponent=fill_exponent,
                                                    fill_multiplier=fill_multiplier,
                                                    step_size=1/n_steps,
                                                    num_trajectories=num_trajectories)
    LOtrader = LimitOrderModelDynamics(midprice_model = midprice_model, arrival_model = arrival_model, 
                                fill_probability_model = fill_probability_model,
                                num_trajectories = num_trajectories)
    reward_function = CjMmCriterion(per_step_inventory_aversion = phi, terminal_inventory_aversion = alpha)
    env_params = dict(terminal_time=terminal_time, 
                      n_steps=n_steps,
                      initial_inventory = initial_inventory,
                      model_dynamics = LOtrader,
                      max_inventory=n_steps,
                      normalise_action_space = False,
                      normalise_observation_space = False,
                      reward_function = reward_function,
                      num_trajectories=num_trajectories)
    return TradingEnvironment(**env_params)